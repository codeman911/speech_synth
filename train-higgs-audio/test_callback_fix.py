#!/usr/bin/env python3
"""
Test script to verify the strategic logging callback fix
"""

import sys
import os
import torch
from datetime import datetime
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerState, TrainerControl

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
        
        # Create mock data that simulates what would be passed from the trainer
        class MockInputs:
            def __init__(self):
                self.input_ids = torch.randint(0, 1000, (2, 50))
                self.attention_mask = torch.ones((2, 50))
                self.audio_features = torch.randn(2, 768, 1500)
                self.audio_in_ids = torch.randint(0, 1024, (8, 100))
                self.audio_waveforms_concat = torch.randn(1, 48000)
                self.label_ids = torch.randint(0, 1000, (2, 50))
                self.label_audio_ids = torch.randint(0, 1024, (2, 8, 100))
        
        class MockOutputs:
            def __init__(self):
                self.logits = torch.randn(2, 50, 1000)
                self.audio_logits = torch.randn(2, 8, 100, 1024)
                self.loss = torch.tensor(2.5)
        
        # Mock state
        class MockState:
            def __init__(self):
                self.global_step = 1
                self.log_history = [{'epoch': 1.0}]
        
        # Mock args
        class MockArgs:
            pass
        
        # Mock control
        control = TrainerControl()
        
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
        
        print("Testing InputLoggerCallback...")
        input_callback.on_step_end(
            args=mock_args,
            state=mock_state,
            control=control,
            inputs=mock_inputs,
            outputs=mock_outputs
        )
        
        print("Testing OutputLoggerCallback...")
        output_callback.on_step_end(
            args=mock_args,
            state=mock_state,
            control=control,
            inputs=mock_inputs,
            outputs=mock_outputs
        )
        
        print("Testing SharedAttentionLoggerCallback...")
        shared_attention_callback.on_step_end(
            args=mock_args,
            state=mock_state,
            control=control,
            inputs=mock_inputs,
            outputs=mock_outputs
        )
        
        print("Testing ZeroShotVerificationLoggerCallback...")
        zero_shot_callback.on_step_end(
            args=mock_args,
            state=mock_state,
            control=control,
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