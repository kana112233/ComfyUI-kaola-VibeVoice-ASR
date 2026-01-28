
import sys
import os
from unittest.mock import MagicMock

# Create a mock for folder_paths before importing nodes
mock_folder_paths = MagicMock()
mock_folder_paths.models_dir = "/tmp/mock_models"
sys.modules["folder_paths"] = mock_folder_paths

# Add current directory to sys.path to ensure local imports work
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"Running tests from: {current_dir}")

try:
    import nodes
    print("✅ Successfully imported nodes.py")
except ImportError as e:
    print(f"❌ Failed to import nodes.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error during nodes.py import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify new classes exist
expected_classes = [
    "VibeVoiceTTSLoader",
    "VibeVoiceTTSInference",
    "VibeVoiceStreamingLoader",
    "VibeVoiceStreamingInference"
]

all_passed = True
for cls_name in expected_classes:
    if hasattr(nodes, cls_name):
        print(f"✅ Found class: {cls_name}")
        # Try calling INPUT_TYPES to check for runtime errors in definition
        try:
            cls = getattr(nodes, cls_name)
            inputs = cls.INPUT_TYPES()
            print(f"   Input Types keys: {list(inputs.get('required', {}).keys())}")
        except Exception as e:
            print(f"   ❌ Error calling INPUT_TYPES for {cls_name}: {e}")
            all_passed = False
    else:
        print(f"❌ Missing class: {cls_name}")
        all_passed = False

if all_passed:
    print("\n🎉 All basic structural tests passed!")
    sys.exit(0)
else:
    print("\n⚠️ Some tests failed.")
    sys.exit(1)
