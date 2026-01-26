import sys
import os
import torch
import numpy as np
import soundfile as sf
import librosa
import folder_paths

# Add VibeVoice_src to system path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
vibevoice_src_path = os.path.join(current_dir, "VibeVoice_src")
if vibevoice_src_path not in sys.path:
    sys.path.append(vibevoice_src_path)

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

class VibeVoiceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("STRING", {"default": "microsoft/VibeVoice-ASR"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "device": (["cuda", "cpu", "mps", "xpu", "auto"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("VIBEVOICE_MODEL",)
    RETURN_NAMES = ("vibevoice_model",)
    FUNCTION = "load_model"
    CATEGORY = "VibeVoice"

    def load_model(self, model_name, precision, device):
        print(f"Loading VibeVoice ASR model: {model_name}")
        
        dtype = torch.float32
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # MPS/CPU usually prefer float32 for stability if not using specific kernels
        if device == "mps" or device == "cpu":
             if precision != "fp32":
                 print(f"Warning: forcing float32 for {device} device stability")
                 dtype = torch.float32

        processor = VibeVoiceASRProcessor.from_pretrained(model_name)
        
        # Load model using from_pretrained
        # We need trust_remote_code=True as per original repo usage
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device if device != "cpu" else None, # accelerate handles device_map
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to(device)

        model.eval()
        
        return ({"model": model, "processor": processor, "device": device, "dtype": dtype},)

class VibeVoiceTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vibevoice_model": ("VIBEVOICE_MODEL",),
                "audio": ("AUDIO",), # ComfyUI AUDIO input
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 65536}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                 "context_info": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter hotwords or context here..."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("srt_content", "json_content")
    FUNCTION = "transcribe"
    CATEGORY = "VibeVoice"

    def transcribe(self, vibevoice_model, audio, max_new_tokens, temperature, top_p, context_info=""):
        model = vibevoice_model["model"]
        processor = vibevoice_model["processor"]
        device = vibevoice_model["device"]
        
        # Audio processing
        # ComfyUI audio format: {"waveform": tensor [batch, channels, samples], "sample_rate": int}
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Take the first batch item and mix to mono if needed
        # VibeVoice expects mono audio
        if waveform.dim() == 3: # [batch, channels, samples]
             waveform = waveform[0] # Take first item in batch
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform_np = waveform.squeeze().cpu().numpy()
        
        # VibeVoice expects 16000Hz usually, let's check processor or default to what the model expects.
        # The demo uses load_audio_use_ffmpeg which resamples to 16000 or 24000 depending on model?
        # Actually standard ASR is often 16k. The repo examples show usage of processor without explicit SR arg sometimes,
        # but processor(audio=..., sampling_rate=...) is standard.
        # Check processor feature extractor sampling rate
        target_sr = processor.feature_extractor.sampling_rate
        
        if sample_rate != target_sr:
             import librosa
             waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=target_sr)
             sample_rate = target_sr

        print(f"Processing audio: {len(waveform_np)} samples at {sample_rate}Hz")
        
        inputs = processor(
            audio=waveform_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info if context_info.strip() else None
        )
        
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
        
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else None,
            "top_p": top_p if (temperature > 0) else None,
            "do_sample": temperature > 0,
            "pad_token_id": processor.pad_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        # remove None values
        generation_config = {k: v for k, v in generation_config.items() if v is not None}

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_config)
            
        # Decode
        input_length = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0, input_length:]
        generated_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        # Post-process
        try:
            segments = processor.post_process_transcription(generated_text)
        except Exception as e:
            print(f"Error parsing segments: {e}")
            segments = []
            
        # Generate SRT
        srt_output = self.generate_srt(segments)
        
        import json
        json_output = json.dumps({"raw_text": generated_text, "segments": segments}, indent=2, ensure_ascii=False)
        
        return (srt_output, json_output)

    def generate_srt(self, segments):
        srt_lines = []
        for i, seg in enumerate(segments):
            start = seg.get('start_time', 0.0)
            end = seg.get('end_time', 0.0)
            text = seg.get('text', '')
            speaker = seg.get('speaker_id', 'Unknown')
            
            # format time: HH:MM:SS,mmm
            def format_time(seconds):
                import datetime
                dt = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=seconds)
                return dt.strftime('%H:%M:%S,%f')[:-3]
            
            srt_lines.append(f"{i+1}")
            srt_lines.append(f"{format_time(start)} --> {format_time(end)}")
            srt_lines.append(f"[{speaker}] {text}")
            srt_lines.append("")
            
        return "\n".join(srt_lines)

NODE_CLASS_MAPPINGS = {
    "VibeVoiceLoader": VibeVoiceLoader,
    "VibeVoiceTranscribe": VibeVoiceTranscribe
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VibeVoiceLoader": "VibeVoice Model Loader",
    "VibeVoiceTranscribe": "VibeVoice Transcribe"
}
