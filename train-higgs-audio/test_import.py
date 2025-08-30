#!/usr/bin/env python3
"""
Test script to verify imports work in the training context
"""

def test_import():
    """Test that we can import the strategic logging callbacks"""
    try:
        # This is how the training script imports the callbacks
        from callbacks.strategic_logging import InputLoggerCallback, OutputLoggerCallback, SharedAttentionLoggerCallback, ZeroShotVerificationLoggerCallback
        print("✅ Import successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_import()
    exit(0 if success else 1)