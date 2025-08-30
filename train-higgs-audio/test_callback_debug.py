#!/usr/bin/env python3
"""
Test script to verify callback debugging information
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_callback_debugging():
    """Test that callback debugging is properly implemented"""
    try:
        # Read the callback file and check for debugging code
        callback_file = os.path.join(project_root, 'callbacks', 'strategic_logging.py')
        
        if not os.path.exists(callback_file):
            print(f"ERROR: Callback file not found at {callback_file}")
            return False
            
        with open(callback_file, 'r') as f:
            content = f.read()
            
        # Check for debugging statements
        debug_checks = [
            "STRATEGIC LOG DEBUG: Received args type",
            "STRATEGIC LOG DEBUG: Received kwargs keys",
            "STRATEGIC LOG DEBUG: No inputs received",
            "STRATEGIC LOG DEBUG OUTPUT:",
            "STRATEGIC LOG DEBUG ZERO-SHOT:"
        ]
        
        missing_debug = []
        for check in debug_checks:
            if check not in content:
                missing_debug.append(check)
                
        if missing_debug:
            print(f"ERROR: Missing debug statements: {missing_debug}")
            return False
            
        print("SUCCESS: Callback debugging is properly implemented")
        print(f"Callback file: {callback_file}")
        print(f"File size: {os.path.getsize(callback_file)} bytes")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_callback_debugging()
    sys.exit(0 if success else 1)