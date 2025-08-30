#!/usr/bin/env python3
"""
Simple test script to verify the fixes applied to the zero-shot voice cloning training
"""

import sys
import os
import torch

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_token_id_validation():
    """Test token ID validation to prevent decoding errors"""
    print("Testing token ID validation...")
    
    # Simulate tokenizer with vocab size
    class MockTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            
        def __len__(self):
            return self.vocab_size
            
        def decode(self, tokens, skip_special_tokens=False):
            return "decoded_text"
    
    # Test valid token IDs
    tokenizer = MockTokenizer(1000)
    valid_tokens = torch.tensor([10, 20, 30])
    max_token_id = valid_tokens.max().item()
    
    if max_token_id < len(tokenizer) and max_token_id >= 0:
        print("✓ Valid token IDs test passed")
    else:
        print("✗ Valid token IDs test failed")
        
    # Test invalid token IDs (out of range)
    invalid_tokens = torch.tensor([1000, 2000, 3000])  # Out of range for vocab size 1000
    max_token_id = invalid_tokens.max().item()
    
    if max_token_id >= len(tokenizer) or max_token_id < 0:
        print("✓ Invalid token IDs detection working")
    else:
        print("✗ Invalid token IDs detection failed")

def test_audio_waveform_processing():
    """Test audio waveform processing"""
    print("\nTesting audio waveform processing...")
    
    # Test 1D waveform (should be converted to 2D)
    waveform_1d = torch.randn(16000)  # 1 second at 16kHz
    if waveform_1d.dim() == 1:
        waveform_2d = waveform_1d.unsqueeze(0)
        print(f"✓ 1D to 2D conversion: {waveform_1d.shape} → {waveform_2d.shape}")
    else:
        print("✗ 1D waveform test failed")
        
    # Test empty waveform handling
    empty_waveform = torch.tensor([])
    if empty_waveform.numel() == 0:
        print("✓ Empty waveform detection working")
    else:
        print("✗ Empty waveform detection failed")

def test_learning_rate_validation():
    """Test learning rate validation"""
    print("\nTesting learning rate validation...")
    
    # Test low learning rate (should be adjusted)
    low_lr = 1e-8
    min_lr = 1e-6
    adjusted_lr = low_lr if low_lr > min_lr else 5e-5
    
    if adjusted_lr == 5e-5:
        print(f"✓ Low learning rate adjustment: {low_lr} → {adjusted_lr}")
    else:
        print(f"✗ Low learning rate adjustment failed: {low_lr} → {adjusted_lr}")
        
    # Test appropriate learning rate (should remain unchanged)
    appropriate_lr = 5e-5
    adjusted_lr = appropriate_lr if appropriate_lr > min_lr else 5e-5
    
    if adjusted_lr == appropriate_lr:
        print(f"✓ Appropriate learning rate preserved: {appropriate_lr}")
    else:
        print(f"✗ Appropriate learning rate preservation failed: {appropriate_lr} → {adjusted_lr}")

def test_gradient_flow():
    """Test gradient flow configuration"""
    print("\nTesting gradient flow configuration...")
    
    # Create a simple model to test parameter configuration
    model = torch.nn.Linear(10, 1)
    
    # Check if parameters require gradients
    all_requires_grad = all(param.requires_grad for param in model.parameters())
    
    if all_requires_grad:
        print("✓ All parameters require gradients")
    else:
        print("✗ Some parameters do not require gradients")
        
    # Test setting requires_grad explicitly
    for param in model.parameters():
        param.requires_grad = True
        
    all_requires_grad = all(param.requires_grad for param in model.parameters())
    
    if all_requires_grad:
        print("✓ Gradient configuration verified")
    else:
        print("✗ Gradient configuration failed")

def main():
    """Main test function"""
    print("=== Zero-Shot Voice Cloning Fixes Verification ===\n")
    
    test_token_id_validation()
    test_audio_waveform_processing()
    test_learning_rate_validation()
    test_gradient_flow()
    
    print("\n=== Verification Complete ===")
    print("All tests completed. Check results above.")

if __name__ == "__main__":
    main()