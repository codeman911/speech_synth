#!/usr/bin/env python3
"""
Test script to verify imports work in the training context
"""

import sys
import os

def test_training_import():
    """Test that we can import the strategic logging callbacks in the training context"""
    try:
        # Add the current directory to the path
        sys.path.insert(0, os.path.dirname(__file__))
        
        # This simulates how the training script imports the callbacks
        import callbacks.strategic_logging
        from callbacks.strategic_logging import InputLoggerCallback, OutputLoggerCallback, SharedAttentionLoggerCallback, ZeroShotVerificationLoggerCallback
        
        print("✅ Training context import successful")
        return True
    except ImportError as e:
        print(f"❌ Training context import failed: {e}")
        return False

def test_callback_creation():
    """Test that we can create callback instances"""
    try:
        # Add the current directory to the path
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Import the callbacks
        from callbacks.strategic_logging import InputLoggerCallback, OutputLoggerCallback, SharedAttentionLoggerCallback, ZeroShotVerificationLoggerCallback
        
        # Try to create instances
        input_callback = InputLoggerCallback()
        output_callback = OutputLoggerCallback()
        shared_attention_callback = SharedAttentionLoggerCallback()
        zero_shot_callback = ZeroShotVerificationLoggerCallback()
        
        print("✅ Callback creation successful")
        return True
    except Exception as e:
        print(f"❌ Callback creation failed: {e}")
        return False

def main():
    print("Testing strategic logging callbacks in training context...\n")
    
    # Test imports
    if not test_training_import():
        return False
    
    # Test callback creation
    if not test_callback_creation():
        return False
    
    print("\n✅ All training context tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)