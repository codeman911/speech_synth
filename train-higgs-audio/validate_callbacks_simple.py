#!/usr/bin/env python3
"""
Simple validation script to check if callbacks can be imported
"""

import sys
import os

def test_callback_import():
    """Test if callbacks can be imported"""
    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        print(f"Project root: {project_root}")
        print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
        
        # Try importing the callbacks
        from callbacks.strategic_logging import (
            InputLoggerCallback, 
            OutputLoggerCallback, 
            SharedAttentionLoggerCallback, 
            ZeroShotVerificationLoggerCallback
        )
        
        print("SUCCESS: All callbacks imported successfully")
        print(f"InputLoggerCallback: {InputLoggerCallback}")
        print(f"OutputLoggerCallback: {OutputLoggerCallback}")
        print(f"SharedAttentionLoggerCallback: {SharedAttentionLoggerCallback}")
        print(f"ZeroShotVerificationLoggerCallback: {ZeroShotVerificationLoggerCallback}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to import callbacks: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_callback_import()
    sys.exit(0 if success else 1)