from .nodes import VibeVoiceLoader, VibeVoiceTranscribe, VibeVoiceSaveFile

NODE_CLASS_MAPPINGS = {
    "VibeVoiceLoader": VibeVoiceLoader,
    "VibeVoiceTranscribe": VibeVoiceTranscribe,
    "VibeVoiceSaveFile": VibeVoiceSaveFile
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VibeVoiceLoader": "VibeVoice Model Loader",
    "VibeVoiceTranscribe": "VibeVoice Transcribe",
    "VibeVoiceSaveFile": "VibeVoice Save File"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
