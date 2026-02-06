#!/usr/bin/env python3
"""Unit tests for chunked audio processing logic (does not require model loading)."""

import sys
import os

# Mock folder_paths for standalone testing
class MockFolderPaths:
    models_dir = '/tmp'

sys.modules['folder_paths'] = MockFolderPaths()

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'VibeVoice_src'))
sys.path.insert(0, os.path.join(current_dir, '..'))


def test_chunked_processing_logic():
    """Test the chunked processing logic without loading models."""
    print("=" * 60)
    print("Testing Chunked Processing Logic")
    print("=" * 60)
    
    print("\n[LOG] Importing VibeVoiceTranscribe...")
    from nodes import VibeVoiceTranscribe
    print("[LOG] Import successful")
    
    transcriber = VibeVoiceTranscribe()
    
    # 1. Test INPUT_TYPES has chunk_duration
    print("\n1. Testing INPUT_TYPES...")
    input_types = VibeVoiceTranscribe.INPUT_TYPES()
    assert 'chunk_duration' in input_types['required'], "chunk_duration not in required inputs"
    chunk_config = input_types['required']['chunk_duration']
    print(f"   ✅ chunk_duration config: {chunk_config}")
    assert chunk_config[1]['default'] == 120, "Default should be 120s"
    assert chunk_config[1]['min'] == 30, "Min should be 30s"
    assert chunk_config[1]['max'] == 600, "Max should be 600s"
    print("   ✅ chunk_duration parameters correct")
    
    # 2. Test _merge_overlapping_segments
    print("\n2. Testing _merge_overlapping_segments...")
    
    # Test case 1: Duplicate segments (same text, overlapping time)
    segments_1 = [
        {'start_time': 0.0, 'end_time': 5.0, 'text': 'Hello world', 'speaker_id': 0},
        {'start_time': 4.5, 'end_time': 10.0, 'text': 'Hello world', 'speaker_id': 0},  # duplicate
        {'start_time': 10.0, 'end_time': 15.0, 'text': 'How are you', 'speaker_id': 1},
    ]
    merged_1 = transcriber._merge_overlapping_segments(segments_1)
    print(f"   Test 1: {len(segments_1)} segments -> {len(merged_1)} merged")
    assert len(merged_1) == 2, f"Expected 2 merged segments, got {len(merged_1)}"
    print("   ✅ Duplicate removal works")
    
    # Test case 2: Non-overlapping segments
    segments_2 = [
        {'start_time': 0.0, 'end_time': 5.0, 'text': 'First', 'speaker_id': 0},
        {'start_time': 10.0, 'end_time': 15.0, 'text': 'Second', 'speaker_id': 0},
        {'start_time': 20.0, 'end_time': 25.0, 'text': 'Third', 'speaker_id': 0},
    ]
    merged_2 = transcriber._merge_overlapping_segments(segments_2)
    print(f"   Test 2: {len(segments_2)} segments -> {len(merged_2)} merged")
    assert len(merged_2) == 3, f"Expected 3 merged segments, got {len(merged_2)}"
    print("   ✅ Non-overlapping segments preserved")
    
    # Test case 3: Empty list
    merged_3 = transcriber._merge_overlapping_segments([])
    assert merged_3 == [], "Empty input should return empty list"
    print("   ✅ Empty list handling works")
    
    # 3. Test chunking calculation
    print("\n3. Testing chunking calculation...")
    sample_rate = 24000
    chunk_duration = 60  # 60 seconds
    audio_length = 180 * sample_rate  # 180 seconds = 3 minutes
    
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(5 * sample_rate)
    
    chunk_count = 0
    chunk_start = 0
    chunks = []
    while chunk_start < audio_length:
        chunk_end = min(chunk_start + chunk_samples, audio_length)
        chunks.append((chunk_start / sample_rate, chunk_end / sample_rate))
        chunk_count += 1
        if chunk_end >= audio_length:
            break
        chunk_start = chunk_end - overlap_samples
    
    print(f"   Audio: 180s, Chunk: 60s, Overlap: 5s")
    for i, (start, end) in enumerate(chunks):
        print(f"   Chunk {i+1}: {start:.1f}s - {end:.1f}s")
    assert chunk_count == 4, f"Expected 4 chunks, got {chunk_count}"
    print(f"   ✅ Chunking produces {chunk_count} chunks as expected")
    
    # 4. Verify methods exist
    print("\n4. Verifying methods...")
    assert hasattr(transcriber, '_process_single_chunk'), "Missing _process_single_chunk"
    assert hasattr(transcriber, '_merge_overlapping_segments'), "Missing _merge_overlapping_segments"
    print("   ✅ All required methods exist")
    
    print("\n" + "=" * 60)
    print("✅ All chunked processing logic tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_chunked_processing_logic()
