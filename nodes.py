import sys
import os
import torch
import numpy as np
import soundfile as sf
import librosa
import folder_paths
import comfy.model_management
from comfy.utils import ProgressBar

# Add VibeVoice_src to system path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
vibevoice_src_path = os.path.join(current_dir, "VibeVoice_src")
if vibevoice_src_path not in sys.path:
    sys.path.append(vibevoice_src_path)

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
import glob
import copy

# VoiceMapper class used in demo
class VoiceMapper:
    """Maps speaker names to voice file paths"""
    
    def __init__(self):
        self.setup_voice_presets()

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        # Adjust path to relative to this file
        voices_dir = os.path.join(current_dir, "VibeVoice_src/demo/voices/streaming_model")
        
        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        # Scan for all VOICE files in the voices directory
        self.voice_presets = {}
        
        # Get all .pt files in the voices directory
        pt_files = glob.glob(os.path.join(voices_dir, "**", "*.pt"), recursive=True)
        
        # Create dictionary with filename (without extension) as key
        for pt_file in pt_files:
            # key: filename without extension
            name = os.path.splitext(os.path.basename(pt_file))[0] # Keep case for display? Demo lowers it.
            # Demo logic:
            # name = os.path.splitext(os.path.basename(pt_file))[0].lower()
            # But let's keep original casing for UI and match case-insensitive
            full_path = os.path.abspath(pt_file)
            self.voice_presets[name] = full_path
        
        self.voice_presets = dict(sorted(self.voice_presets.items()))

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        # First try exact match
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        
        # Try case-insensitive
        speaker_name_lower = speaker_name.lower()
        for name, path in self.voice_presets.items():
            if name.lower() == speaker_name_lower:
                 return path
        
        # Try partial matching (case insensitive)
        matched_path = None
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_name_lower or speaker_name_lower in preset_name.lower():
                if matched_path is not None:
                     # Ambiguous, but let's just pick one
                     pass 
                matched_path = path
        if matched_path is not None:
            return matched_path
        
        # Default to first voice if no match found
        if self.voice_presets:
             default_voice = list(self.voice_presets.values())[0]
             print(f"Warning: No voice preset found for '{speaker_name}', using default voice: {default_voice}")
             return default_voice
        return None

class VibeVoiceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("STRING", {"default": "microsoft/VibeVoice-ASR", "tooltip": "HuggingFace repo ID or local path to model"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16", "tooltip": "Model precision. Use fp16 for VRAM saving"}),
                "device": (["cuda", "cpu", "mps", "xpu", "auto"], {"default": "auto", "tooltip": "Device to run the model on"}),
            },
        }

    RETURN_TYPES = ("VIBEVOICE_MODEL",)
    RETURN_NAMES = ("vibevoice_model",)
    FUNCTION = "load_model"
    CATEGORY = "VibeVoice"

    def load_model(self, model_name, precision, device):
        print(f"Loading VibeVoice ASR model: {model_name}")

        # Path resolution logic
        model_path = model_name
        if not os.path.exists(model_path):
            # Try looking in ComfyUI/models
            comfy_models_dir = folder_paths.models_dir
            search_paths = [
                os.path.join(comfy_models_dir, model_name),
                os.path.join(comfy_models_dir, "vibevoice", model_name),
                # Check ../models (sibling of this repo) as a fallback for manual downloads
                os.path.abspath(os.path.join(current_dir, "../models", model_name)),
                os.path.abspath(os.path.join(current_dir, "../models", "vibevoice", model_name)),
            ]
            
            # If model_name has a slash (e.g. microsoft/VibeVoice-ASR), try checking just the name (VibeVoice-ASR)
            if "/" in model_name:
                model_basename = model_name.split("/")[-1]
                search_paths.append(os.path.join(comfy_models_dir, model_basename))
                search_paths.append(os.path.join(comfy_models_dir, "vibevoice", model_basename))
                search_paths.append(os.path.abspath(os.path.join(current_dir, "../models", model_basename)))
                search_paths.append(os.path.abspath(os.path.join(current_dir, "../models", "vibevoice", model_basename)))
            
            for potential_path in search_paths:
                if os.path.exists(potential_path):
                    model_path = potential_path
                    break
        
        # If it's a local path, use absolute path to avoid ambiguity
        if os.path.exists(model_path):
            model_path = os.path.abspath(model_path)
            print(f"Resolved model path to: {model_path}")
        else:
            print(f"Model path not found locally, assuming HuggingFace repo ID: {model_name}")
            model_path = model_name

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

        # CPU usually prefers float32 for stability
        # MPS can use fp16 to save memory (model is stored in bf16/fp16)
        if device == "cpu":
            if precision != "fp32":
                print(f"Warning: forcing float32 for {device} device stability")
                dtype = torch.float32

        processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        
        # Load model using from_pretrained
        # We need trust_remote_code=True as per original repo usage
        # low_cpu_mem_usage=True allows loading large models with limited memory
        # For MPS: load to CPU first then move to MPS to avoid memory pre-allocation issues
        if device == "mps":
            print("Loading model to CPU first, then moving to MPS...")
            model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                model_path,
                dtype=dtype,
                device_map=None,  # Load to CPU first
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.to("mps")
        elif device == "cuda":
            model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                model_path,
                dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:  # CPU
            model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                model_path,
                dtype=dtype,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        model.eval()
        
        return ({"model": model, "processor": processor, "device": device, "dtype": dtype},)

class VibeVoiceTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vibevoice_model": ("VIBEVOICE_MODEL", {"tooltip": "Loaded VibeVoice model"}),
                "audio": ("AUDIO", {"tooltip": "Input audio to transcribe"}), # ComfyUI AUDIO input
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 65536, "tooltip": "Max tokens to generate"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Randomness (0.0 = deterministic)"}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling probability"}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Penalty for repetition (>1.0 reduces repetition)"}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "Beam search width (1 = greedy)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed"}),
                "chunk_duration": ("INT", {"default": 120, "min": 10, "max": 600, "step": 1, "tooltip": "Split audio into chunks of this duration (seconds) to save memory"}),
            },
            "optional": {
                 "context_info": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter hotwords or context here...", "tooltip": "Context prompt / hotwords"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("srt_content", "json_content", "raw_text", "speaker_log")
    FUNCTION = "transcribe"
    CATEGORY = "VibeVoice"

    def transcribe(self, vibevoice_model, audio, max_new_tokens, temperature, top_p, repetition_penalty, num_beams, seed, chunk_duration=120, context_info=""):
        if seed is not None:
             torch.manual_seed(seed)
             if torch.cuda.is_available():
                 torch.cuda.manual_seed_all(seed)
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
        
        # VibeVoice usually uses 24000Hz or 16000Hz
        # Try to get sampling rate from attributes, fallback to 16000
        target_sr = 16000
        if hasattr(processor, "target_sample_rate"):
             target_sr = processor.target_sample_rate
        elif hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "sampling_rate"):
            target_sr = processor.feature_extractor.sampling_rate
        elif hasattr(processor, "sampling_rate"):
            target_sr = processor.sampling_rate
        
        print(f"DEBUG - Target sampling rate: {target_sr}, Input sampling rate: {sample_rate}")
        
        if sample_rate != target_sr:
             print(f"DEBUG - Resampling from {sample_rate}Hz to {target_sr}Hz")
             import librosa
             waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=target_sr)
             sample_rate = target_sr
        else:
             print("DEBUG - Sample rates match. Skipping resampling.")

        audio_duration = len(waveform_np) / sample_rate
        print(f"Processing audio: {len(waveform_np)} samples at {sample_rate}Hz ({audio_duration:.1f}s)")
        
        # Chunked processing for long audio to avoid OOM
        # If audio is longer than chunk_duration, split into overlapping chunks
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(5 * sample_rate)  # 5 second overlap to avoid cutting words
        
        if len(waveform_np) > chunk_samples:
            print(f"\n{'='*60}")
            print(f"Long audio detected ({audio_duration:.1f}s > {chunk_duration}s)")
            print(f"Using chunked processing with {chunk_duration}s chunks")
            print(f"{'='*60}\n")
            
            all_segments = []
            all_raw_texts = []
            chunk_start = 0
            # Calculate total chunks for progress bar
            total_chunks = int(np.ceil((len(waveform_np)) / (chunk_samples - overlap_samples)))
            pbar = ProgressBar(total_chunks)
            
            chunk_start = 0
            chunk_idx = 0
            
            while chunk_start < len(waveform_np):
                # Check for cancellation
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                chunk_end = min(chunk_start + chunk_samples, len(waveform_np))
                chunk_audio = waveform_np[chunk_start:chunk_end]
                
                # Calculate time offset for this chunk
                time_offset = chunk_start / sample_rate
                
                print(f"\n[Chunk {chunk_idx + 1}/{total_chunks}] Processing {time_offset:.1f}s - {chunk_end/sample_rate:.1f}s ({len(chunk_audio)/sample_rate:.1f}s)")
                
                # Process this chunk
                try:
                    chunk_segments, chunk_raw_text = self._process_single_chunk(
                        chunk_audio, sample_rate, processor, model, device,
                        max_new_tokens, temperature, top_p, repetition_penalty,
                        num_beams, context_info, time_offset
                    )
                    
                    all_segments.extend(chunk_segments)
                    all_raw_texts.append(chunk_raw_text)
                    print(f"  ✓ Found {len(chunk_segments)} segments in this chunk")
                    
                except Exception as e:
                    # Check if it's an interrupt exception just in case it was caught
                    if "interrupted" in str(e).lower():
                        raise e
                        
                    print(f"  ✗ Error processing chunk: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Update progress bar
                pbar.update(1)

                # Clear GPU memory after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                # Move to next chunk (with overlap handling)
                if chunk_end >= len(waveform_np):
                    break
                chunk_start = chunk_end - overlap_samples
                chunk_idx += 1
            
            # Merge overlapping segments (remove duplicates from overlap regions)
            segments = self._merge_overlapping_segments(all_segments, overlap_duration=5.0)
            generated_text = "\n\n===CHUNK SEPARATOR===\n\n".join(all_raw_texts)
            
            print(f"\n{'='*60}")
            print(f"Chunked processing complete: {len(segments)} total segments")
            print(f"{'='*60}\n")
        else:
            # Standard processing for short audio
            segments, generated_text = self._process_single_chunk(
                waveform_np, sample_rate, processor, model, device,
                max_new_tokens, temperature, top_p, repetition_penalty,
                num_beams, context_info, time_offset=0.0
            )
        
        # Generate SRT
        srt_output = self.generate_srt(segments)
        speaker_log = self.generate_log(segments)
        
        import json
        json_output = json.dumps({"raw_text": generated_text, "segments": segments}, indent=2, ensure_ascii=False)
        
        return (srt_output, json_output, generated_text, speaker_log)
    
    def _process_single_chunk(self, waveform_np, sample_rate, processor, model, device,
                               max_new_tokens, temperature, top_p, repetition_penalty,
                               num_beams, context_info, time_offset=0.0):
        """Process a single audio chunk and return segments with adjusted timestamps."""
        
        inputs = processor(
            audio=waveform_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info if context_info.strip() else None
        )
        
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
        
        do_sample = temperature > 0
        
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
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
        # If empty, try decoding with special tokens to see what happened
        if not generated_text.strip():
            generated_text_raw = processor.decode(generated_ids, skip_special_tokens=False)
            print(f"Warning: Generated text was empty. Raw output with special tokens: {repr(generated_text_raw)}")
            if generated_text_raw.strip():
                generated_text = generated_text_raw # Fallback to raw text if standard decode is empty
        print(f"DEBUG: Chunk generated text length: {len(generated_text)}, content repr: {repr(generated_text)}")
        
        # Post-process
        try:
            segments = processor.post_process_transcription(generated_text)
        except Exception as e:
            print(f"Error parsing segments: {e}")
            segments = []  # Will be handled below

        # Check for empty segments (could be due to parsing error caught inside processor)
        if not segments and generated_text.strip():
            print(f"Warning: No segments found but generated text exists. Attempting to repair/fallback.")
            # print(f"Raw generated text causing issue: {generated_text}") # Reduced verbosity as we might fix it
            
            # Try to repair truncated JSON
            try:
                # Find start of JSON
                json_start = generated_text.find("[")
                if json_start != -1:
                    potential_json = generated_text[json_start:]
                    # Try to parse as is (unlikely if processor failed, but worth a shot if logic differed)
                    try:
                        segments = json.loads(potential_json)
                    except json.JSONDecodeError:
                        # Common case: truncated output. Try appending ']' or '}]' or '"}]'
                        # Simple repair: try finding the last valid object end '}'
                        last_object_end = potential_json.rfind("}")
                        if last_object_end != -1:
                            # Construct a valid array up to the last object
                            repaired_json = potential_json[:last_object_end+1] + "]"
                            try:
                                segments = json.loads(repaired_json)
                                print(f"  ✓ Successfully repaired truncated JSON! Recovered segments.")
                            except:
                                pass

            except Exception as e:
                print(f"  ✗ JSON repair failed: {e}")

            # If still empty, use the raw text fallback
            if not segments:
                print(f"  Using raw text fallback.")
                segments = [{
                    "start_time": 0, 
                    "end_time": 0, 
                    "text": f"[JSON_ERROR] {generated_text}", 
                    "speaker_id": 0
                }]
        
        # Adjust timestamps by adding time_offset
        if time_offset > 0:
            for seg in segments:
                if 'start_time' in seg:
                    seg['start_time'] = seg['start_time'] + time_offset
                if 'end_time' in seg:
                    seg['end_time'] = seg['end_time'] + time_offset
        
        return segments, generated_text
    
    def _merge_overlapping_segments(self, all_segments, overlap_duration=5.0):
        """Merge segments from overlapping chunks, removing duplicates."""
        if not all_segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(all_segments, key=lambda x: x.get('start_time', 0))
        
        merged = []
        for seg in sorted_segments:
            if not merged:
                merged.append(seg)
                continue
            
            last_seg = merged[-1]
            last_end = last_seg.get('end_time', 0)
            curr_start = seg.get('start_time', 0)
            
            # Check if this segment overlaps with the previous one
            # and has similar text (likely a duplicate from overlap region)
            if curr_start < last_end:
                # These segments overlap - check text similarity
                last_text = last_seg.get('text', '').strip()
                curr_text = seg.get('text', '').strip()
                
                # If texts are very similar or one contains the other, skip the duplicate
                if last_text == curr_text:
                    continue
                if len(last_text) > 10 and len(curr_text) > 10:
                    # Check if significant portion overlaps (simple heuristic)
                    if last_text[-20:] == curr_text[:20] or last_text in curr_text or curr_text in last_text:
                        continue
            
            merged.append(seg)
        
        return merged

    def generate_log(self, segments):
        if not isinstance(segments, list):
            print(f"Warning: segments is not a list: {type(segments)}")
            return ""

        log_lines = []
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue
                
            start = seg.get('start_time', 0.0)
            end = seg.get('end_time', 0.0)
            text = seg.get('text', '')
            speaker = seg.get('speaker_id', 'Unknown')
            
            # format time: HH:MM:SS,mmm
            def format_time(seconds):
                import datetime
                # Handle potential float precision making huge seconds
                try:
                    dt = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=seconds)
                    return dt.strftime('%H:%M:%S,%f')[:-3]
                except:
                    return f"{seconds:.2f}"
            
            # Custom format: Index Start --> End speaker: text
            line = f"{i+1} {format_time(start)} --> {format_time(end)} speaker{speaker}: {text}"
            log_lines.append(line)
            
        return "\n\n".join(log_lines)

    def generate_srt(self, segments, speaker_prefix=False):
        if not isinstance(segments, list):
            return ""

        srt_lines = []
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue

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
            if speaker_prefix:
                srt_lines.append(f"speaker{speaker}: {text}")
            else:
                srt_lines.append(f"[{speaker}] {text}")
            srt_lines.append("")
            
        return "\n".join(srt_lines)

class VibeVoiceShowText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show_text"
    CATEGORY = "VibeVoice"
    OUTPUT_NODE = True

    def show_text(self, text):
        print(f"####################\n[VibeVoiceShowText] Content:\n{text}\n####################")
        return {"ui": {"text": [text]}, "result": (text,)}

class VibeVoiceTTSLoader:
    @classmethod
    def INPUT_TYPES(s):
        # Discover available models in vibevoice directory
        available_models = ["microsoft/VibeVoice-1.5B"]  # Default HuggingFace option
        
        # Search for local models
        try:
            vibevoice_dir = os.path.join(folder_paths.models_dir, "vibevoice")
            if os.path.isdir(vibevoice_dir):
                for item in os.listdir(vibevoice_dir):
                    item_path = os.path.join(vibevoice_dir, item)
                    # Check for checkpoints subdirectory (VibeVoice-1.5B structure)
                    checkpoints_path = os.path.join(item_path, "checkpoints")
                    if os.path.isdir(checkpoints_path) and os.path.exists(os.path.join(checkpoints_path, "config.json")):
                        available_models.append(f"{item}/checkpoints")
                    # Check for direct model directory (VibeVoice-0.5B structure)
                    elif os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                        available_models.append(item)
        except Exception as e:
            print(f"Error scanning vibevoice models: {e}")
        
        return {
            "required": {
                "model_name": (available_models, {"default": available_models[0], "tooltip": "HuggingFace repo ID or local path"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16", "tooltip": "Model precision"}),
                "device": (["cuda", "cpu", "mps", "auto"], {"default": "auto", "tooltip": "Device"}),
            },
        }

    RETURN_TYPES = ("VIBEVOICE_TTS_MODEL",)
    RETURN_NAMES = ("vibevoice_tts_model",)
    FUNCTION = "load_model"
    CATEGORY = "VibeVoice"

    def load_model(self, model_name, precision, device):
        print(f"Loading VibeVoice TTS model: {model_name}")
        
        # Path resolution logic (similar to ASR)
        model_path = model_name
        if not os.path.exists(model_path):
             comfy_models_dir = folder_paths.models_dir
             search_paths = [
                 os.path.join(comfy_models_dir, model_name),
                 os.path.join(comfy_models_dir, "vibevoice", model_name),
                 os.path.abspath(os.path.join(current_dir, "../models", model_name)),
                 os.path.abspath(os.path.join(current_dir, "../models", "vibevoice", model_name)),
             ]
             # Handle model names like "VibeVoice-1.5B/checkpoints"
             if "/" in model_name:
                 model_basename = model_name.split("/")[-1]
                 model_parent = model_name.rsplit("/", 1)[0] if "/" in model_name else model_name
                 search_paths.append(os.path.join(comfy_models_dir, model_basename))
                 search_paths.append(os.path.join(comfy_models_dir, "vibevoice", model_basename))
                 # Full path with parent directory (e.g., vibevoice/VibeVoice-1.5B/checkpoints)
                 search_paths.insert(0, os.path.join(comfy_models_dir, "vibevoice", model_name))
                 search_paths.append(os.path.abspath(os.path.join(current_dir, "../models", model_basename)))
                 search_paths.append(os.path.abspath(os.path.join(current_dir, "../models", "vibevoice", model_basename)))
             
             for potential_path in search_paths:
                 if os.path.exists(potential_path):
                     # Check if this path has config.json or if checkpoints subdirectory has it
                     if os.path.exists(os.path.join(potential_path, "config.json")):
                         model_path = potential_path
                         break
                     # Auto-detect checkpoints subdirectory
                     checkpoints_subdir = os.path.join(potential_path, "checkpoints")
                     if os.path.exists(os.path.join(checkpoints_subdir, "config.json")):
                         model_path = checkpoints_subdir
                         break
        
        if os.path.exists(model_path):
             model_path = os.path.abspath(model_path)
             print(f"Resolved model path to: {model_path}")
        else:
             print(f"Model path not found locally, assuming HuggingFace repo ID: {model_name}")
             model_path = model_name

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

        # MPS/CPU stability check
        if device == "mps" or device == "cpu":
             if precision != "fp32":
                 print(f"Warning: forcing float32 for {device} device stability")
                 dtype = torch.float32

        # Load Processor
        processor = VibeVoiceProcessor.from_pretrained(model_path)
        
        # Load Model
        # Use Inference class which contains generate method
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to(device)

        # Ensure lm_head shares weights with embed_tokens (tie_word_embeddings=true in config)
        model.tie_weights()
        model.eval()
        
        return ({"model": model, "processor": processor, "device": device, "dtype": dtype},)

class VibeVoiceTTSInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vibevoice_tts_model": ("VIBEVOICE_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Speaker 1: Hello world."}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 65536}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}), # Classifier-Free Guidance
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "reference_audio": ("AUDIO",), # Optional reference audio for speaker cloning/conditioning
                "speaker_indices": ("STRING", {"default": "0", "placeholder": "Example: 0, 1 (Map to provided reference audios if list)"}), 
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "VibeVoice"

    def generate(self, vibevoice_tts_model, text, max_new_tokens, temperature, cfg_scale, seed, reference_audio=None, speaker_indices="0"):
        if seed is not None:
             torch.manual_seed(seed)
             if torch.cuda.is_available():
                 torch.cuda.manual_seed_all(seed)
        
        model = vibevoice_tts_model["model"]
        processor = vibevoice_tts_model["processor"]
        device = vibevoice_tts_model["device"]
        dtype = vibevoice_tts_model["dtype"]

        # Handle reference audio
        voice_samples = []
        if reference_audio is not None:
            waveform = reference_audio["waveform"]
            # mix to mono
            if waveform.dim() == 3:
                waveform = waveform[0]
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform_np = waveform.squeeze().cpu().numpy()
            
            # Resample if needed (Processor usually expects 24k or 16k depending on config)
            # We'll use the processor's audio_processor sampling rate
            target_sr = 24000
            if hasattr(processor, "audio_processor") and hasattr(processor.audio_processor, "sampling_rate"):
                target_sr = processor.audio_processor.sampling_rate
            
            input_sr = reference_audio["sample_rate"]
            if input_sr != target_sr:
                waveform_np = librosa.resample(waveform_np, orig_sr=input_sr, target_sr=target_sr)
            
            voice_samples.append(waveform_np)
        
        # Auto-format text to meet processor requirements if needed
        # Processor expects: "Speaker N: Text"
        import re
        if not re.match(r"Speaker\s+\d+\s*:", text, re.IGNORECASE):
             print(f"Details: Text auto-formatted to 'Speaker 0: {text[:20]}...'")
             text = f"Speaker 0: {text}"

        # Prepare inputs
        inputs = processor(
            text=text,
            voice_samples=voice_samples if voice_samples else None,
            return_tensors="pt",
        )
        
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
        
        # Generation config
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "cfg_scale": cfg_scale, 
        }



        print(f"Generating TTS audio for text: {text[:50]}...")
        
        with torch.no_grad():
             outputs = model.generate(
                 **inputs,
                 **generation_config,
                 tokenizer=processor.tokenizer, # Needed for some generation paths
             )

        # Extract audio
        # outputs.speech_outputs is likely a list of waveforms
        if hasattr(outputs, "speech_outputs") and outputs.speech_outputs and len(outputs.speech_outputs) > 0 and outputs.speech_outputs[0] is not None:
            generated_audio = outputs.speech_outputs[0].cpu().float()
            
            # Create ComfyUI audio output
            # [batch, channels, samples] -> [1, 1, samples]
            output_audio = generated_audio.unsqueeze(0).unsqueeze(0)
            
            return ({"waveform": output_audio, "sample_rate": 24000},) # VibeVoice default is 24k usually
        else:
            print("No audio generated.")
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000},)


class VibeVoiceTTSInferenceMultiSpeaker:
    """TTS Inference node with support for multiple speaker reference audios"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vibevoice_tts_model": ("VIBEVOICE_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Speaker 0: Hello. Speaker 1: Hi there."}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 65536}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "speaker_0_audio": ("AUDIO",), # Reference audio for Speaker 0
                "speaker_1_audio": ("AUDIO",), # Reference audio for Speaker 1
                "speaker_2_audio": ("AUDIO",), # Reference audio for Speaker 2
                "speaker_3_audio": ("AUDIO",), # Reference audio for Speaker 3
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "VibeVoice"

    def generate(self, vibevoice_tts_model, text, max_new_tokens, temperature, cfg_scale, seed,
                 speaker_0_audio=None, speaker_1_audio=None, speaker_2_audio=None, speaker_3_audio=None):
        if seed is not None:
             torch.manual_seed(seed)
             if torch.cuda.is_available():
                 torch.cuda.manual_seed_all(seed)

        model = vibevoice_tts_model["model"]
        processor = vibevoice_tts_model["processor"]
        device = vibevoice_tts_model["device"]
        dtype = vibevoice_tts_model["dtype"]

        # Map input audios by speaker ID
        speaker_audios = [speaker_0_audio, speaker_1_audio, speaker_2_audio, speaker_3_audio]

        # Parse text to extract speaker segments
        # Support both single-line and multi-line formats:
        # - Single line: "Speaker 0: Hello. Speaker 1: Hi."
        # - Multi-line: "Speaker 0: Hello.\nSpeaker 1: Hi."
        import re

        # Find all "Speaker N:" markers and their positions
        pattern = r'Speaker\s+(\d+)\s*:'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))

        if not matches:
            # No speaker markers found, treat entire text as Speaker 0
            print("Warning: No Speaker markers found, treating as Speaker 0")
            segments = [(0, text.strip())]
        else:
            segments = []
            for i, match in enumerate(matches):
                speaker_id = int(match.group(1))
                start_pos = match.end()  # Position after "Speaker N:"

                # Find end position (start of next Speaker or end of text)
                if i < len(matches) - 1:
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(text)

                # Extract text content
                text_content = text[start_pos:end_pos].strip()

                if text_content:  # Skip empty segments
                    segments.append((speaker_id, text_content))
                    print(f"Parsed: Speaker {speaker_id}: {text_content[:50]}...")

        print(f"\n{'='*60}")
        print(f"Multi-Speaker TTS - Separate Generation Mode")
        print(f"Total segments: {len(segments)}")
        print(f"{'='*60}\n")

        # Determine target sample rate
        target_sr = 24000
        if hasattr(processor, "audio_processor") and hasattr(processor.audio_processor, "sampling_rate"):
            target_sr = processor.audio_processor.sampling_rate

        # Process reference audios
        processed_audios = {}
        for idx in range(len(speaker_audios)):
            if speaker_audios[idx] is not None:
                waveform = speaker_audios[idx]["waveform"]
                # mix to mono
                if waveform.dim() == 3:
                    waveform = waveform[0]
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                waveform_np = waveform.squeeze().cpu().numpy()

                # Resample if needed
                input_sr = speaker_audios[idx]["sample_rate"]
                if input_sr != target_sr:
                    waveform_np = librosa.resample(waveform_np, orig_sr=input_sr, target_sr=target_sr)

                processed_audios[idx] = waveform_np
                print(f"✓ Speaker {idx}: Reference audio loaded ({len(waveform_np)} samples)")

        # Generate audio for each segment separately
        all_audio_segments = []

        for seg_idx, (speaker_id, text_content) in enumerate(segments):
            print(f"\n[Segment {seg_idx + 1}/{len(segments)}] Speaker {speaker_id}: {text_content[:50]}...")

            # Prepare single-speaker text with Speaker 0 format (for consistent processing)
            single_speaker_text = f"Speaker 0: {text_content}"

            # Get reference audio for this speaker
            voice_samples = None
            if speaker_id in processed_audios:
                voice_samples = [processed_audios[speaker_id]]
                print(f"  Using reference audio for Speaker {speaker_id}")
            else:
                print(f"  Using default voice (no reference audio for Speaker {speaker_id})")

            # Prepare inputs
            inputs = processor(
                text=single_speaker_text,
                voice_samples=voice_samples,
                return_tensors="pt",
            )

            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}

            # Generation config
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "cfg_scale": cfg_scale,
            }

            # Generate audio for this segment
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_config,
                    tokenizer=processor.tokenizer,
                )

            # Extract audio
            if hasattr(outputs, "speech_outputs") and outputs.speech_outputs and len(outputs.speech_outputs) > 0 and outputs.speech_outputs[0] is not None:
                segment_audio = outputs.speech_outputs[0].cpu().float()

                # Ensure 1D tensor: squeeze all extra dimensions
                while segment_audio.dim() > 1:
                    segment_audio = segment_audio.squeeze(0)

                # Verify we got valid audio
                if segment_audio.numel() > 0:
                    all_audio_segments.append(segment_audio)
                    print(f"  ✓ Generated {segment_audio.shape[0]} samples ({segment_audio.shape[0]/target_sr:.2f}s)")
                else:
                    print(f"  ✗ Warning: Empty audio generated for this segment")
            else:
                print(f"  ✗ Warning: No audio generated for this segment")

        # Concatenate all audio segments
        if not all_audio_segments:
            print("\nError: No audio segments were generated")
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": target_sr},)

        # Add short silence between segments (100ms)
        silence_duration = int(target_sr * 0.1)
        silence = torch.zeros(silence_duration)

        final_audio_parts = []
        for i, segment in enumerate(all_audio_segments):
            final_audio_parts.append(segment)
            # Add silence between segments (but not after the last one)
            if i < len(all_audio_segments) - 1:
                final_audio_parts.append(silence)

        final_audio = torch.cat(final_audio_parts, dim=0)

        print(f"\n{'='*60}")
        print(f"Generation Complete:")
        print(f"  Total segments: {len(all_audio_segments)}")
        print(f"  Final audio length: {len(final_audio)} samples ({len(final_audio)/target_sr:.2f}s)")
        print(f"{'='*60}\n")

        # Format for ComfyUI: [batch, channels, samples]
        output_audio = final_audio.unsqueeze(0).unsqueeze(0)
        return ({"waveform": output_audio, "sample_rate": target_sr},)




class VibeVoiceStreamingLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("STRING", {"default": "microsoft/VibeVoice-Realtime-0.5B", "tooltip": "HuggingFace repo ID or local path"}),
                "device": (["cuda", "cpu", "mps", "auto"], {"default": "auto", "tooltip": "Device to run the model on"}),
            },
        }

    RETURN_TYPES = ("VIBEVOICE_STREAMING_MODEL",)
    RETURN_NAMES = ("vibevoice_streaming_model",)
    FUNCTION = "load_model"
    CATEGORY = "VibeVoice"

    def load_model(self, model_name, device):
        print(f"Loading VibeVoice Streaming model: {model_name}")
        
        model_path = model_name
        # Same path resolution
        if not os.path.exists(model_path):
             comfy_models_dir = folder_paths.models_dir
             search_paths = [
                 os.path.join(comfy_models_dir, model_name),
                 os.path.join(comfy_models_dir, "vibevoice", model_name),
                 os.path.abspath(os.path.join(current_dir, "../models", model_name)),
                 os.path.abspath(os.path.join(current_dir, "../models", "vibevoice", model_name)),
             ]
             if "/" in model_name:
                 model_basename = model_name.split("/")[-1]
                 search_paths.append(os.path.join(comfy_models_dir, model_basename))
                 search_paths.append(os.path.join(comfy_models_dir, "vibevoice", model_basename))
                 search_paths.append(os.path.abspath(os.path.join(current_dir, "../models", model_basename)))
                 search_paths.append(os.path.abspath(os.path.join(current_dir, "../models", "vibevoice", model_basename)))
             for p in search_paths:
                 if os.path.exists(p):
                     model_path = p
                     break
        
        if os.path.exists(model_path):
             model_path = os.path.abspath(model_path)

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Dtype logic as per demo
        attn_metrics = "sdpa"
        if device == "cuda":
            dtype = torch.bfloat16
            attn_metrics = "flash_attention_2"
        elif device == "mps":
            dtype = torch.float32
        else:
            dtype = torch.float32

        processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
        
        # Load Model
        try:
             model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device if device != "cpu" else None,
                attn_implementation=attn_metrics,
                trust_remote_code=True,
                low_cpu_mem_usage=False # Prevent meta tensor issues on CPU
            )
        except Exception as e:
            print(f"Failed to load with {attn_metrics}, falling back to SDPA. Error: {e}")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device if device != "cpu" else None,
                attn_implementation="sdpa",
                trust_remote_code=True,
                low_cpu_mem_usage=False # Prevent meta tensor issues on CPU
            )
            
        if device == "mps":
            model = model.to("mps")
            
        model.eval()
        model.set_ddpm_inference_steps(num_steps=5) # Default from demo

        return ({"model": model, "processor": processor, "device": device, "dtype": dtype},)

class VibeVoiceStreamingInference:
    @classmethod
    def INPUT_TYPES(s):
        # Scan for presets
        presets = ["Wayne"]
        try:
            voices_dir = os.path.join(current_dir, "VibeVoice_src", "demo", "voices", "streaming_model")
            if os.path.exists(voices_dir):
                files = glob.glob(os.path.join(voices_dir, "**", "*.pt"), recursive=True)
                presets = sorted([os.path.splitext(os.path.basename(f))[0] for f in files])
        except Exception as e:
            print(f"Error scanning presets: {e}")

        return {
            "required": {
                "vibevoice_streaming_model": ("VIBEVOICE_STREAMING_MODEL", {"tooltip": "Loaded streaming model"}),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a real-time streaming test.", "tooltip": "Text to generate speech from"}),
                "speaker_name": (presets, {"default": presets[0] if presets else "", "tooltip": "Voice preset to use"}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Classifier Free Guidance scale"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "VibeVoice"

    def generate(self, vibevoice_streaming_model, text, speaker_name, cfg_scale, seed):
        if seed is not None:
             torch.manual_seed(seed)
             if torch.cuda.is_available():
                 torch.cuda.manual_seed_all(seed)
        
        model = vibevoice_streaming_model["model"]
        processor = vibevoice_streaming_model["processor"]
        device = vibevoice_streaming_model["device"]
        # dtype = vibevoice_streaming_model["dtype"]

        print(f"Generating Streaming TTS for: '{text}' with speaker '{speaker_name}'")

        # 1. initialize voice mapper
        voice_mapper = VoiceMapper()
        voice_path = voice_mapper.get_voice_path(speaker_name)
        
        if not voice_path or not os.path.exists(voice_path):
             raise ValueError(f"Voice preset not found for: {speaker_name}")

        print(f"Using voice preset: {voice_path}")
        
        # 2. Load cached prompt
        target_device = device if device != "cpu" else "cpu"
        all_prefilled_outputs = torch.load(voice_path, map_location=target_device, weights_only=False)

        # 3. Process Input
        full_script = text.replace("’", "'").replace('“', '"').replace('”', '"')
        
        inputs = processor.process_input_with_cached_prompt(
            text=full_script,
            cached_prompt=all_prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)

        # 4. Generate
        print(f"Starting generation with cfg_scale: {cfg_scale}")
        
        # Ensure deepcopy if needed, similar to demo
        import copy
        prefilled_copy = copy.deepcopy(all_prefilled_outputs) if all_prefilled_outputs is not None else None

        with torch.no_grad():
             outputs = model.generate(
                 **inputs,
                 max_new_tokens=None, # handled by generate logic internally or default
                 cfg_scale=cfg_scale,
                 tokenizer=processor.tokenizer,
                 generation_config={'do_sample': False},
                 verbose=True,
                 all_prefilled_outputs=prefilled_copy,
             )

        # 5. Extract audio
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            # Output is [1, Length] usually (mono)
            generated_audio = outputs.speech_outputs[0].cpu().float()
            
            # ComfyUI expects [batch, channels, samples] or [batch, samples, channels]?
            # Actually standard AUDIO type is {"waveform": tensor[batch, channels, samples], "sample_rate": int}
            
            # generated_audio is [Length] or [1, Length]
            if generated_audio.dim() == 1:
                generated_audio = generated_audio.unsqueeze(0) # [1, Length]
            
            # Add batch dim -> [1, 1, Length]
            waveform = generated_audio.unsqueeze(0)
            
            return ({"waveform": waveform, "sample_rate": 24000},) 
        else:
            print("No audio output generated")
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000},)
    CATEGORY = "VibeVoice"

    def generate(self, vibevoice_streaming_model, text, speaker_name, cfg_scale, seed):
        if seed is not None:
             torch.manual_seed(seed)
             if torch.cuda.is_available():
                 torch.cuda.manual_seed_all(seed)

        model = vibevoice_streaming_model["model"]
        processor = vibevoice_streaming_model["processor"]
        device = vibevoice_streaming_model["device"]
        
        # Load preset directly
        voice_path = None
        voices_dir = os.path.join(current_dir, "VibeVoice_src", "demo", "voices", "streaming_model")
        # Find the file matching speaker_name
        if os.path.exists(voices_dir):
            files = glob.glob(os.path.join(voices_dir, "**", "*.pt"), recursive=True)
            for f in files:
                if os.path.splitext(os.path.basename(f))[0] == speaker_name:
                    voice_path = f
                    break
        
        if not voice_path or not os.path.exists(voice_path):
             print(f"Warning: Preset {speaker_name} not found. Trying to find any pt file.")
             if files:
                 voice_path = files[0]
                 print(f"Using fallback: {voice_path}")
             else:
                 raise ValueError(f"No voice preset found for {speaker_name} and no fallbacks available.")

        print(f"Loading voice preset from {voice_path}")
        target_device = device if device != "cpu" else "cpu"
        all_prefilled_outputs = torch.load(voice_path, map_location=target_device, weights_only=False)

        # Process inputs
        full_script = text.replace("’", "'").replace('“', '"').replace('”', '"')
        
        inputs = processor.process_input_with_cached_prompt(
            text=full_script,
            cached_prompt=all_prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
        
        print(f"Generating Streaming Audio for: {text[:50]}...")
        with torch.no_grad():
             outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False},
                # verbose=True,
                all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs)
            )
            
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
             # Streaming model usually outputs 24kHz too
             audio_tensor = outputs.speech_outputs[0] # [samples]
             output_audio = audio_tensor.unsqueeze(0).unsqueeze(0).cpu().float()
             return ({"waveform": output_audio, "sample_rate": 24000},)
        else:
             return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000},)

class VibeVoiceSaveFile:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "vibevoice_output"}),
                "file_extension": (["srt", "txt", "json"], {"default": "srt"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_file"
    OUTPUT_NODE = True
    CATEGORY = "VibeVoice"

    def save_file(self, text, filename_prefix="vibevoice_output", file_extension="srt"):
        import time
        # Use simple timestamp-based naming to avoid overwrite
        # If user wants subfolders, they can include slash in prefix, but we need to handle directory creation
        
        # Handle subfolders in prefix
        if "/" in filename_prefix or "\\" in filename_prefix:
            # Normalize slashes
            filename_prefix = filename_prefix.replace("\\", "/")
            subfolder = os.path.dirname(filename_prefix)
            base_prefix = os.path.basename(filename_prefix)
            full_output_dir = os.path.join(self.output_dir, subfolder)
            if not os.path.exists(full_output_dir):
                os.makedirs(full_output_dir)
        else:
            base_prefix = filename_prefix
            full_output_dir = self.output_dir
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{base_prefix}_{timestamp}.{file_extension}"
        file_path = os.path.join(full_output_dir, filename)
        
        with open(file_path, "w", encoding="utf-8-sig") as f:
            f.write(text)
            
        print(f"Saved text to: {file_path}")
        return {
            "ui": {
                "text": [f"Saved to: {file_path}"],
                "file_info": [{
                    "filename": filename,
                    "subfolder": subfolder if "/" in filename_prefix or "\\" in filename_prefix else "",
                    "type": "output"
                }]
            }, 
            "result": (file_path,)
        }

NODE_CLASS_MAPPINGS = {
    "VibeVoiceLoader": VibeVoiceLoader,
    "VibeVoiceTranscribe": VibeVoiceTranscribe,
    "VibeVoiceShowText": VibeVoiceShowText,
    "VibeVoiceSaveFile": VibeVoiceSaveFile,
    "VibeVoiceTTSLoader": VibeVoiceTTSLoader,
    "VibeVoiceTTSInference": VibeVoiceTTSInference,
    "VibeVoiceTTSInferenceMultiSpeaker": VibeVoiceTTSInferenceMultiSpeaker,
    "VibeVoiceStreamingLoader": VibeVoiceStreamingLoader,
    "VibeVoiceStreamingInference": VibeVoiceStreamingInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VibeVoiceLoader": "VibeVoice Model Loader (ASR)",
    "VibeVoiceTranscribe": "VibeVoice Transcribe (ASR)",
    "VibeVoiceShowText": "VibeVoice Show String",
    "VibeVoiceSaveFile": "VibeVoice Save File",
    "VibeVoiceTTSLoader": "VibeVoice TTS Model Loader",
    "VibeVoiceTTSInference": "VibeVoice TTS Inference",
    "VibeVoiceTTSInferenceMultiSpeaker": "VibeVoice TTS Inference (Multi-Speaker)",
    "VibeVoiceStreamingLoader": "VibeVoice Streaming Model Loader",
    "VibeVoiceStreamingInference": "VibeVoice Streaming Inference",
}
