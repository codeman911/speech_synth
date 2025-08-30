#!/usr/bin/env python3
"""
Debug script to test if callbacks are being called
"""

import sys
import os
from datetime import datetime

# Add the trainer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

def test_callback_trigger():
    """Test callback trigger logic"""
    print("Testing callback trigger logic...")
    
    # Simulate different steps
    log_every_n_steps = 100
    steps = [1, 50, 100, 150, 200, 250, 300]
    
    for step in steps:
        # This is the logic in our callbacks
        if step % log_every_n_steps != 0:
            print(f"Step {step}: NOT logging (condition: {step} % {log_every_n_steps} = {step % log_every_n_steps} != 0)")
        else:
            print(f"Step {step}: SHOULD LOG (condition: {step} % {log_every_n_steps} = {step % log_every_n_steps} == 0)")
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_callback_trigger()