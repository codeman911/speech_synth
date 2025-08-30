#!/usr/bin/env python3
"""
Test script to verify callback structure without requiring dependencies
"""

import sys
import os

def test_callback_structure():
    """Test callback structure by parsing the file"""
    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        callbacks_path = os.path.join(project_root, 'callbacks', 'strategic_logging.py')
        
        if not os.path.exists(callbacks_path):
            print(f"ERROR: Callbacks file not found at {callbacks_path}")
            return False
            
        # Read the file and check for class definitions
        with open(callbacks_path, 'r') as f:
            content = f.read()
            
        required_classes = [
            'InputLoggerCallback',
            'OutputLoggerCallback', 
            'SharedAttentionLoggerCallback',
            'ZeroShotVerificationLoggerCallback'
        ]
        
        missing_classes = []
        for class_name in required_classes:
            if f'class {class_name}' not in content:
                missing_classes.append(class_name)
                
        if missing_classes:
            print(f"ERROR: Missing classes: {missing_classes}")
            return False
            
        print("SUCCESS: All required callback classes found in strategic_logging.py")
        print(f"File size: {os.path.getsize(callbacks_path)} bytes")
        print(f"Last modified: {os.path.getmtime(callbacks_path)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_callback_structure()
    sys.exit(0 if success else 1)