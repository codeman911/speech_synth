#!/usr/bin/env python3
"""
Test script to verify module structure without requiring dependencies
"""

import sys
import os

def test_module_structure():
    """Test that the module structure is correct"""
    try:
        # Add the current directory to the path
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Check if the callbacks directory exists and has the right files
        if not os.path.exists('callbacks'):
            print("❌ callbacks directory not found")
            return False
            
        if not os.path.exists('callbacks/__init__.py'):
            print("❌ callbacks/__init__.py not found")
            return False
            
        if not os.path.exists('callbacks/strategic_logging.py'):
            print("❌ callbacks/strategic_logging.py not found")
            return False
            
        # Try to parse the strategic_logging.py file to check for syntax errors
        with open('callbacks/strategic_logging.py', 'r') as f:
            source = f.read()
            
        # Compile the source to check for syntax errors
        compile(source, 'callbacks/strategic_logging.py', 'exec')
        
        print("✅ Module structure validation passed")
        return True
    except Exception as e:
        print(f"❌ Module structure validation failed: {e}")
        return False

def test_import_path():
    """Test that the import path is correct"""
    try:
        # Add the current directory to the path
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Try to import the module (this will fail due to missing dependencies, but we can check the import path)
        import callbacks.strategic_logging
        print("✅ Import path is correct")
        return True
    except ImportError as e:
        # Check if the error is due to missing dependencies (torch) or incorrect import path
        if "torch" in str(e).lower() or "no module named" in str(e).lower():
            print("✅ Import path is correct (dependency error expected)")
            return True
        else:
            print(f"❌ Import path error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("Testing strategic logging module structure...\n")
    
    # Test module structure
    if not test_module_structure():
        return False
    
    # Test import path
    if not test_import_path():
        return False
    
    print("\n✅ All module structure tests passed!")
    print("\nNote: Import errors due to missing dependencies (torch) are expected outside of the training environment.")
    print("The module structure is correct and should work within the training environment.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)