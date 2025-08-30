#!/usr/bin/env python3
"""
Test script to verify collator structure without requiring dependencies
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_collator_structure():
    """Test the collator structure"""
    try:
        # Read the collator code and check for key components
        trainer_file = os.path.join(project_root, 'trainer', 'train_v2_ddp.py')
        
        if not os.path.exists(trainer_file):
            print(f"ERROR: Trainer file not found at {trainer_file}")
            return False
            
        with open(trainer_file, 'r') as f:
            content = f.read()
            
        # Check for key components in the fallback implementation
        checks = [
            ("Fallback implementation", "Fallback implementation"),
            ("audio_waveforms_concat", "audio_waveforms_concat"),
            ("audio_sample_rate", "audio_sample_rate"),
            ("audio_speaker_indices", "audio_speaker_indices"),
            ("ExtendedHiggsAudioBatchInput", "ExtendedHiggsAudioBatchInput"),
        ]
        
        failed_checks = []
        for check_name, check_content in checks:
            if check_content not in content:
                failed_checks.append(check_name)
                
        if failed_checks:
            print(f"ERROR: Failed checks: {failed_checks}")
            return False
            
        print("SUCCESS: Collator structure is correct")
        print(f"Trainer file: {trainer_file}")
        print(f"File size: {os.path.getsize(trainer_file)} bytes")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_collator_structure()
    sys.exit(0 if success else 1)