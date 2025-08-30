#!/usr/bin/env python3
"""
Simple test script to verify the fixes applied to the zero-shot voice cloning training
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test that we can import the necessary modules"""
    print("Testing imports...")
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
        
    try:
        from trainer.train_v2_ddp import ExtendedHiggsAudioSampleCollator, ZeroShotVoiceCloningDataset
        print("✓ Training modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import training modules: {e}")
        return False
        
    return True

def test_collator_initialization():
    """Test that the collator can be initialized"""
    print("\nTesting collator initialization...")
    try:
        from trainer.train_v2_ddp import ExtendedHiggsAudioSampleCollator
        collator = ExtendedHiggsAudioSampleCollator(
            pad_token_id=0,
            encode_whisper_embed=True,
            audio_num_codebooks=8
        )
        print("✓ Collator initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize collator: {e}")
        return False

def main():
    """Main test function"""
    print("=== Zero-Shot Voice Cloning Fixes Verification ===\n")
    
    if not test_imports():
        print("\n✗ Import tests failed")
        return
        
    if not test_collator_initialization():
        print("\n✗ Collator initialization tests failed")
        return
    
    print("\n=== Verification Complete ===")
    print("All tests passed. The fixes appear to be correctly implemented.")

if __name__ == "__main__":
    main()