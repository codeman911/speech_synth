#!/usr/bin/env python3
"""
Test script for strategic logging callbacks
"""

import torch
from transformers import AutoTokenizer
from trainer.strategic_logging_callbacks import (
    InputLoggerCallback, 
    OutputLoggerCallback, 
    SharedAttentionLoggerCallback, 
    ZeroShotVerificationLoggerCallback
)

def test_input_logger():
    """Test the InputLoggerCallback"""
    print("Testing InputLoggerCallback...")
    
    # Mock model inputs
    class MockModelInputs:
        def __init__(self):
            self.input_ids = torch.randint(0, 1000, (2, 50))
            self.attention_mask = torch.ones((2, 50))
            self.audio_features = torch.randn(2, 768, 1500)
            self.audio_in_ids = torch.randint(0, 1024, (8, 100))
            self.audio_waveforms_concat = torch.randn(1, 48000)
    
    # Mock state
    class MockState:
        def __init__(self):
            self.global_step = 100
            self.log_history = [{'epoch': 1.0}]
    
    # Mock args
    class MockArgs:
        pass
    
    # Initialize callback
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    callback = InputLoggerCallback(tokenizer=tokenizer, log_every_n_steps=50)
    
    # Test the callback
    mock_inputs = MockModelInputs()
    mock_state = MockState()
    mock_args = MockArgs()
    
    callback.on_log(
        args=mock_args,
        state=mock_state,
        control=None,
        logs={},
        inputs=mock_inputs
    )
    
    print("InputLoggerCallback test completed.\n")

def test_output_logger():
    """Test the OutputLoggerCallback"""
    print("Testing OutputLoggerCallback...")
    
    # Mock model outputs
    class MockModelOutputs:
        def __init__(self):
            self.logits = torch.randn(2, 50, 1000)
            self.audio_logits = torch.randn(2, 8, 100, 1024)
    
    # Mock model inputs
    class MockModelInputs:
        def __init__(self):
            self.label_ids = torch.randint(0, 1000, (2, 50))
            self.label_audio_ids = torch.randint(0, 1024, (2, 8, 100))
    
    # Mock state
    class MockState:
        def __init__(self):
            self.global_step = 100
            self.log_history = [{'epoch': 1.0}]
    
    # Mock args
    class MockArgs:
        pass
    
    # Initialize callback
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    callback = OutputLoggerCallback(tokenizer=tokenizer, log_every_n_steps=50)
    
    # Test the callback
    mock_outputs = MockModelOutputs()
    mock_inputs = MockModelInputs()
    mock_state = MockState()
    mock_args = MockArgs()
    
    callback.on_log(
        args=mock_args,
        state=mock_state,
        control=None,
        logs={'loss': 2.5, 'grad_norm': 0.3},
        outputs=mock_outputs,
        inputs=mock_inputs
    )
    
    print("OutputLoggerCallback test completed.\n")

def test_shared_attention_logger():
    """Test the SharedAttentionLoggerCallback"""
    print("Testing SharedAttentionLoggerCallback...")
    
    # Mock model
    class MockModelConfig:
        def __init__(self):
            self.audio_dual_ffn_layers = [2, 5, 8, 11]
    
    class MockTextModel:
        def __init__(self):
            self.layers = [None] * 12  # Simulate 12 layers
            self.config = MockModelConfig()
    
    class MockHiggsModel:
        def __init__(self):
            self.text_model = MockTextModel()
            self.config = MockModelConfig()
    
    class MockModel:
        def __init__(self):
            self.model = MockHiggsModel()
    
    # Mock state
    class MockState:
        def __init__(self):
            self.global_step = 100
            self.log_history = [{'epoch': 1.0}]
    
    # Mock args
    class MockArgs:
        pass
    
    # Initialize callback
    callback = SharedAttentionLoggerCallback(log_every_n_steps=50)
    
    # Test the callback
    mock_model = MockModel()
    mock_state = MockState()
    mock_args = MockArgs()
    
    callback.on_log(
        args=mock_args,
        state=mock_state,
        control=None,
        logs={},
        model=mock_model
    )
    
    print("SharedAttentionLoggerCallback test completed.\n")

def test_zero_shot_verification_logger():
    """Test the ZeroShotVerificationLoggerCallback"""
    print("Testing ZeroShotVerificationLoggerCallback...")
    
    # Mock model inputs
    class MockModelInputs:
        def __init__(self):
            self.input_ids = torch.randint(0, 1000, (2, 50))
            self.audio_features = torch.randn(2, 768, 1500)
            self.audio_in_ids = torch.randint(0, 1024, (8, 100))
    
    # Mock model config
    class MockModelConfig:
        def __init__(self):
            self.encode_whisper_embed = True
            self.audio_in_token_idx = 128015
            self.audio_out_token_idx = 128016
    
    # Mock model
    class MockHiggsModel:
        def __init__(self):
            self.config = MockModelConfig()
    
    class MockModel:
        def __init__(self):
            self.model = MockHiggsModel()
    
    # Mock state
    class MockState:
        def __init__(self):
            self.global_step = 100
            self.log_history = [{'epoch': 1.0}]
    
    # Mock args
    class MockArgs:
        pass
    
    # Initialize callback
    callback = ZeroShotVerificationLoggerCallback(log_every_n_steps=50)
    
    # Test the callback
    mock_inputs = MockModelInputs()
    mock_model = MockModel()
    mock_state = MockState()
    mock_args = MockArgs()
    
    callback.on_log(
        args=mock_args,
        state=mock_state,
        control=None,
        logs={},
        inputs=mock_inputs,
        model=mock_model
    )
    
    print("ZeroShotVerificationLoggerCallback test completed.\n")

if __name__ == "__main__":
    print("Running strategic logging callback tests...\n")
    
    try:
        test_input_logger()
        test_output_logger()
        test_shared_attention_logger()
        test_zero_shot_verification_logger()
        
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()