
import sys
import os
import unittest
from unittest.mock import MagicMock

# Mock folder_paths which is used by nodes.py
mock_folder_paths = MagicMock()
mock_folder_paths.models_dir = "/tmp/mock_models"
sys.modules["folder_paths"] = mock_folder_paths

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import nodes
try:
    import nodes
except ImportError:
    # If standard import fails (e.g. missing dependencies), we might need to mock more things
    # But let's check if we can just mock vibevoice if it's missing
    mock_vibevoice = MagicMock()
    sys.modules["vibevoice"] = mock_vibevoice
    sys.modules["vibevoice.processor"] = MagicMock()
    sys.modules["vibevoice.processor.vibevoice_streaming_processor"] = MagicMock()
    import nodes

class TestVibeVoiceTranscribe(unittest.TestCase):
    def test_return_types(self):
        print("\nTesting VibeVoiceTranscribe RETURN_TYPES...")
        node = nodes.VibeVoiceTranscribe()
        self.assertEqual(len(node.RETURN_TYPES), 4, "Should have 4 return types")
        self.assertEqual(node.RETURN_NAMES[-1], "srt_with_speaker", "Last return name should be srt_with_speaker")
        print("✅ RETURN_TYPES verified")

    def test_generate_srt_speaker_prefix(self):
        print("\nTesting generate_srt with speaker prefix...")
        node = nodes.VibeVoiceTranscribe()
        
        segments = [
            {"start_time": 0.0, "end_time": 2.5, "text": "Hello world", "speaker_id": 1},
            {"start_time": 3.0, "end_time": 4.0, "text": "Foo bar", "speaker_id": 0}
        ]
        
        # Test Default (Old format)
        srt_default = node.generate_srt(segments)
        print(f"Default Output Sample:\n{srt_default[:100]}...")
        self.assertIn("[1] Hello world", srt_default)
        self.assertIn("[0] Foo bar", srt_default)
        
        # Test New Format
        srt_new = node.generate_srt(segments, speaker_prefix=True)
        print(f"New Output Sample:\n{srt_new[:100]}...")
        self.assertIn("speaker1: Hello world", srt_new)
        self.assertIn("speaker0: Foo bar", srt_new)
        
        print("✅ generate_srt formats verified")

if __name__ == "__main__":
    unittest.main()
