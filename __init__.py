from .nodes import VibeVoiceLoader, VibeVoiceTranscribe

NODE_CLASS_MAPPINGS = {
    "VibeVoiceLoader": VibeVoiceLoader,
    "VibeVoiceTranscribe": VibeVoiceTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VibeVoiceLoader": "VibeVoice Model Loader",
    "VibeVoiceTranscribe": "VibeVoice Transcribe",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
