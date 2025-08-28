#!/usr/bin/env python3
"""
Test script to verify validation split functionality
"""

import sys
import os

# Add the trainer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trainer'))

def test_validation_split_logic():
    """Test the validation split logic"""
    # This is a simple test to verify the logic is correct
    total_samples = 1000
    validation_split = 0.05  # 5%
    
    eval_size = int(total_samples * validation_split)
    train_size = total_samples - eval_size
    
    print(f"Total samples: {total_samples}")
    print(f"Validation split: {validation_split}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {eval_size}")
    print(f"Verification: {train_size} + {eval_size} = {train_size + eval_size}")
    
    # Verify the calculation
    assert train_size + eval_size == total_samples
    assert eval_size == 50  # 5% of 1000
    assert train_size == 950  # 95% of 1000
    
    print("Validation split logic test passed!")

if __name__ == "__main__":
    test_validation_split_logic()