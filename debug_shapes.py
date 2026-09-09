import torch
import numpy as np
from transformers import AutoProcessor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "VibeVoice_src"))

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

# Load the processor and model
model_name = "/home/yons/work/ai/ComfyUI/models/vibevoice/VibeVoice-ASR"
processor = VibeVoiceASRProcessor.from_pretrained(model_name)
model = VibeVoiceASRForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True
)

# Mock audio input
sample_rate = 24000
waveform_np = np.zeros(205079, dtype=np.float32)

print("Processor running...")
inputs = processor(
    audio=waveform_np,
    sampling_rate=sample_rate,
    return_tensors="pt",
    add_generation_prompt=True,
)

print(f"inputs['input_ids'] shape: {inputs['input_ids'].shape if 'input_ids' in inputs else 'None'}")
print(f"inputs['speech_tensors'] shape: {inputs.get('speech_tensors').shape}")
print(f"inputs['acoustic_input_mask'] shape: {inputs.get('acoustic_input_mask').shape}")
print(f"inputs['speech_masks'] shape: {inputs.get('speech_masks').shape}")

print("\nRunning model.generate() to replicate crash...")
try:
    outputs = model.generate(
        **inputs,
        max_new_tokens=10
    )
    print("Model generation succeeded!")
except Exception as e:
    print(f"Error during model generation: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
