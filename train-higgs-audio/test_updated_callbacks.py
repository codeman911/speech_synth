#!/usr/bin/env python3
"""
Test script to verify the updated callback logic
"""

import sys
import os
from datetime import datetime

# Add the trainer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

def test_updated_callback_trigger():
    """Test updated callback trigger logic"""
    print("Testing updated callback trigger logic...")
    
    # Simulate different steps
    log_every_n_steps = 100
    steps = [1, 50, 100, 150, 200, 250, 300]
    
    for step in steps:
        # This is the updated logic in our callbacks
        if step % log_every_n_steps != 0 and step != 1:
            print(f"Step {step}: NOT logging (condition: {step} % {log_every_n_steps} = {step % log_every_n_steps} != 0 AND {step} != 1)")
        else:
            print(f"Step {step}: SHOULD LOG (condition: {step} % {log_every_n_steps} = {step % log_every_n_steps} == 0 OR {step} == 1)")
    
    print("\nTest completed.")
    print("With this update, you should see logs at step 1 and every 100 steps (100, 200, 300, etc.)")

if __name__ == "__main__":
    test_updated_callback_trigger()