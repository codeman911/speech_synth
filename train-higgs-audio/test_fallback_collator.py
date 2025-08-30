#!/usr/bin/env python3
"""
Test script to verify fallback collator implementation
"""

import sys
import os
import torch

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_fallback_collator():
    """Test the fallback collator implementation"""
    try:
        # Import the collator
        from trainer.train_v2_ddp import ExtendedHiggsAudioSampleCollator, ExtendedHiggsAudioBatchInput
        
        # Create a mock dataset sample with audio data
        class MockSample:
            def __init__(self):
                self.input_ids = torch.tensor([1, 2, 3, 4, 5])
                self.label_ids = torch.tensor([1, 2, 3, 4, 5])
                self.audio_waveforms_concat = torch.tensor([0.1, 0.2, 0.3])
                self.audio_waveforms_start = torch.tensor([0])
                self.audio_sample_rate = torch.tensor([24000.0])
                self.audio_speaker_indices = torch.tensor([0])
        
        # Create collator with fallback
        collator = ExtendedHiggsAudioSampleCollator(pad_token_id=0)
        
        # Create a batch of mock samples
        batch = [MockSample(), MockSample()]
        
        # Test the collator
        result = collator(batch)
        
        print("SUCCESS: Fallback collator executed successfully")
        print(f"Result type: {type(result)}")
        print(f"Has input_ids: {hasattr(result, 'input_ids')}")
        print(f"Has audio_waveforms_concat: {hasattr(result, 'audio_waveforms_concat')}")
        
        if hasattr(result, 'input_ids'):
            print(f"input_ids shape: {result.input_ids.shape}")
            
        if hasattr(result, 'audio_waveforms_concat'):
            print(f"audio_waveforms_concat: {result.audio_waveforms_concat}")
            
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fallback_collator()
    sys.exit(0 if success else 1)