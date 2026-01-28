
import sys
import os
import torch
from unittest.mock import MagicMock

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock ComfyUI folder_paths
sys.modules["folder_paths"] = MagicMock()
sys.modules["folder_paths"].models_dir = "/tmp/mock_models"

from nodes import VibeVoiceTTSLoader, VibeVoiceTTSInference

def test_1_5b_tts():
    print("--- Testing Restored VibeVoice 1.5B TTS ---")
    
    loader = VibeVoiceTTSLoader()
    inference = VibeVoiceTTSInference()
    
    # Path found in user's downloads
    model_name = "/Users/xiohu/Downloads/VibeVoice-1.5B/VibeVoice-1.5B/checkpoints"
    
    try:
        if not os.path.exists(model_name):
            print(f"❌ Model path not found: {model_name}")
            return

        # 1. Load Model
        print(f"1. Loading model from: {model_name}")
        # explicit cpu for test safety
        result = loader.load_model(model_name, precision="fp32", device="cpu") 
        model_pack = result[0]
        model = model_pack["model"] # This should now be VibeVoiceForConditionalGenerationInference
        
        print("✅ Model loaded successfully.")
        print(f"   Model Class: {type(model)}")
        
        # 2. Generate Audio
        text = "This is a test of the restored 1.5B model."
        print(f"2. Generating audio for: '{text}'")
        
        # Mock inputs mimicking ComfyUI flow
        audio_output = inference.generate(
            vibevoice_tts_model=model_pack,
            text=text,
            max_new_tokens=500,
            temperature=0.7,
            cfg_scale=1.5,
            seed=42,
            reference_audio=None, # Optional
            speaker_indices="0"
        )
        
        # 3. Validation
        wav = audio_output[0]["waveform"]
        sr = audio_output[0]["sample_rate"]
        print(f"   Generated Audio Shape: {wav.shape}, Sample Rate: {sr}")
        
        if wav.numel() > 0 and wav.shape[-1] > 1000:
            print("✅ Audio generation successful! The 1.5B model is WORKING.")
        else:
            print("❌ Audio generation produced empty output.")

    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_1_5b_tts()
