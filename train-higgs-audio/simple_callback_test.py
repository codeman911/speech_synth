#!/usr/bin/env python3
"""
Simple test script to verify the strategic logging callback fix without requiring torch
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_callback_fix():
    """Test that the callback fix works correctly"""
    try:
        # Import the callbacks
        from callbacks.strategic_logging import (
            InputLoggerCallback, 
            OutputLoggerCallback, 
            SharedAttentionLoggerCallback, 
            ZeroShotVerificationLoggerCallback
        )
        
        print("Testing strategic logging callback fix...")
        
        # Create simple mock data
        class MockInputs:
            def __init__(self):
                self.input_ids = [1, 2, 3, 4, 5]
        
        class MockOutputs:
            def __init__(self):
                self.loss = 2.5
        
        # Mock state
        class MockState:
            def __init__(self):
                self.global_step = 1
                self.log_history = [{'epoch': 1.0}]
        
        # Mock args
        class MockArgs:
            pass
        
        # Mock control
        class MockControl:
            pass
        
        # Initialize callbacks
        input_callback = InputLoggerCallback(log_every_n_steps=1)
        output_callback = OutputLoggerCallback(log_every_n_steps=1)
        shared_attention_callback = SharedAttentionLoggerCallback(log_every_n_steps=1)
        zero_shot_callback = ZeroShotVerificationLoggerCallback(log_every_n_steps=1)
        
        # Create mock data
        mock_inputs = MockInputs()
        mock_outputs = MockOutputs()
        mock_state = MockState()
        mock_args = MockArgs()
        mock_control = MockControl()
        
        print("Testing InputLoggerCallback...")
        # This should not raise an exception
        input_callback.on_step_end(
            args=mock_args,
            state=mock_state,
            control=mock_control,
            inputs=mock_inputs,
            outputs=mock_outputs
        )
        
        print("Testing OutputLoggerCallback...")
        # This should not raise an exception
        output_callback.on_step_end(
            args=mock_args,
            state=mock_state,
            control=mock_control,
            inputs=mock_inputs,
            outputs=mock_outputs
        )
        
        print("Testing SharedAttentionLoggerCallback...")
        # This should not raise an exception
        shared_attention_callback.on_step_end(
            args=mock_args,
            state=mock_state,
            control=mock_control,
            inputs=mock_inputs,
            outputs=mock_outputs
        )
        
        print("Testing ZeroShotVerificationLoggerCallback...")
        # This should not raise an exception
        zero_shot_callback.on_step_end(
            args=mock_args,
            state=mock_state,
            control=mock_control,
            inputs=mock_inputs,
            outputs=mock_outputs
        )
        
        print("\n✅ All callbacks executed successfully!")
        print("The fix should now allow strategic logging to work properly during training.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing callback fix: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_callback_fix()
    sys.exit(0 if success else 1)