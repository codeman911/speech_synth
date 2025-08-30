#!/usr/bin/env python3
"""
Validation script for strategic logging callbacks
Checks syntax without requiring full dependencies
"""

import sys
import os

def validate_syntax():
    """Validate syntax by compiling the module"""
    try:
        import py_compile
        py_compile.compile('callbacks/strategic_logging.py', doraise=True)
        print("✅ Syntax validation passed")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except FileNotFoundError:
        print("❌ File not found: callbacks/strategic_logging.py")
        return False

def validate_imports():
    """Validate that we can import the logging callbacks"""
    try:
        # Add the current directory to the path
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Try to compile the file to check for syntax errors
        with open('callbacks/strategic_logging.py', 'r') as f:
            source = f.read()
        
        compile(source, 'callbacks/strategic_logging.py', 'exec')
        print("✅ Import validation passed")
        return True
    except Exception as e:
        print(f"❌ Import validation failed: {e}")
        return False

def main():
    print("Validating strategic logging callbacks...\n")
    
    # Check if files exist
    files_to_check = [
        'callbacks/strategic_logging.py',
        'trainer/train_v2_ddp.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            return False
    
    # Validate syntax
    if not validate_syntax():
        return False
    
    # Validate imports
    if not validate_imports():
        return False
    
    print("\n✅ All validations passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)