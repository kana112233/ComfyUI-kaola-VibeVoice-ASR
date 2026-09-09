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

# Manually run the first pass of the model to trace where the shape mismatch happens!
print("\nRunning model forward pass to intercept inputs_embeds...")
try:
    outputs = model(
        input_ids=inputs.get('input_ids'),
        speech_tensors=inputs.get('speech_tensors'),
        speech_masks=inputs.get('speech_masks'),
        acoustic_input_mask=inputs.get('acoustic_input_mask'),
        output_hidden_states=False,
    )
    print("Model forward pass succeeded!")
except Exception as e:
    print(f"Error during model forward: {type(e).__name__}: {str(e)}")
    
    # Let's inspect the shapes of the tensors directly
    print("\nTracing manual embedding preparation...")
    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
    print(f"Initial inputs_embeds shape: {inputs_embeds.shape}")
    
    speech_features = model.encode_speech(
        speech_tensors=inputs['speech_tensors'],
        speech_masks=inputs.get('speech_masks'),
        speech_semantic_tensors=None,
    )
    print(f"speech_features shape: {speech_features.shape}")
    
    try:
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[inputs['acoustic_input_mask']] = speech_features
        print(f"inputs_embeds shape after assignment: {inputs_embeds.shape}")
    except Exception as assign_e:
        print(f"Failed to assign speech_features to inputs_embeds: {assign_e}")
