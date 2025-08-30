#!/usr/bin/env python3
"""
Simple test to understand how the callback system works
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

class TestCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        print(f"TestCallback.on_step_end called with kwargs: {kwargs.keys() if kwargs else 'None'}")
        print(f"  args type: {type(args)}")
        print(f"  state type: {type(state)}")
        print(f"  control type: {type(control)}")
        for key, value in kwargs.items():
            print(f"  {key}: {type(value)} = {value}")

def test_callback_system():
    """Test the callback system"""
    print("Testing callback system...")
    
    # Create callback handler
    callback = TestCallback()
    
    # Create dummy objects
    args = TrainingArguments(output_dir="./test")
    state = TrainerState()
    control = TrainerControl()
    
    # Call the callback directly
    print("\n1. Calling callback with no extra kwargs:")
    callback.on_step_end(args, state, control)
    
    # Call the callback with some data
    print("\n2. Calling callback with sample data:")
    sample_data = {"test_key": "test_value"}
    callback.on_step_end(args, state, control, **sample_data)

if __name__ == "__main__":
    test_callback_system()