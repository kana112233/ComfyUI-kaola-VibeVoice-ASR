#!/usr/bin/env python3
"""Test script for VibeVoice ASR (SRT) node."""

import sys
import os

# Mock folder_paths for standalone testing
class MockFolderPaths:
    models_dir = '/Users/xiohu/work/ai-tools/models'

sys.modules['folder_paths'] = MockFolderPaths()

# Add VibeVoice_src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'VibeVoice_src'))
sys.path.insert(0, os.path.join(current_dir, '..'))

import torch
import numpy as np

def test_asr():
    print("=" * 60)
    print("Testing VibeVoice ASR Node")
    print("=" * 60)
    
    # Import nodes
    print("\n[LOG] Importing nodes...")
    from nodes import VibeVoiceLoader, VibeVoiceTranscribe
    print("[LOG] Nodes imported successfully")
    
    # 1. Load ASR model
    print("\n1. Loading VibeVoice ASR model...")
    loader = VibeVoiceLoader()
    
    try:
        print("[LOG] Calling loader.load_model()...")
        result = loader.load_model(
            model_name="/Users/xiohu/work/ai-tools/models/VibeVoice-ASR",
            precision="fp16",  # Use fp16 to save memory
            device="mps"
        )
        model_bundle = result[0]
        print(f"✅ ASR model loaded successfully.")
        print(f"   Model type: {type(model_bundle['model'])}")
    except Exception as e:
        print(f"❌ Failed to load ASR model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Create test audio (1 second of silence)
    print("\n2. Creating test audio...")
    sample_rate = 24000
    duration = 1.0  # seconds
    samples = int(sample_rate * duration)
    
    # Create a simple audio tensor (sine wave)
    t = np.linspace(0, duration, samples)
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz tone
    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # [1, samples]
    
    test_audio = {
        "waveform": audio_tensor.unsqueeze(0),  # [batch, channels, samples]
        "sample_rate": sample_rate
    }
    print(f"   Audio shape: {test_audio['waveform'].shape}, Sample rate: {sample_rate}")
    
    # 3. Test transcription
    print("\n3. Testing transcription...")
    transcriber = VibeVoiceTranscribe()
    
    try:
        print("[LOG] Calling transcriber.transcribe()...")
        srt_content, json_content, raw_text, speaker_log = transcriber.transcribe(
            vibevoice_model=model_bundle,
            audio=test_audio,
            max_new_tokens=4096,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            num_beams=1,
            seed=0,
            chunk_duration=120,  # New parameter
            context_info=""
        )
        print(f"✅ Transcription successful!")
        print(f"   SRT Content: {srt_content[:100]}..." if len(srt_content) > 100 else f"   SRT Content: {srt_content}")
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✅ ASR test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_asr()

