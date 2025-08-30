#!/usr/bin/env python3
"""
Validation script for strategic logging callbacks
Checks syntax and imports without requiring full dependencies
"""

import sys
import os

# Add the trainer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

def validate_imports():
    """Validate that we can import the logging callbacks"""
    try:
        from strategic_logging_callbacks import (
            InputLoggerCallback, 
            OutputLoggerCallback, 
            SharedAttentionLoggerCallback, 
            ZeroShotVerificationLoggerCallback
        )
        print("✅ All logging callbacks imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def validate_syntax():
    """Validate syntax by compiling the module"""
    try:
        import py_compile
        py_compile.compile('trainer/strategic_logging_callbacks.py', doraise=True)
        print("✅ Syntax validation passed")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except FileNotFoundError:
        print("❌ File not found: trainer/strategic_logging_callbacks.py")
        return False

def main():
    print("Validating strategic logging callbacks...\n")
    
    # Check if files exist
    files_to_check = [
        'trainer/strategic_logging_callbacks.py',
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
    print("\nTo use strategic logging in training, run:")
    print("  python train_v2_ddp.py \\")
    print("    --model_path /path/to/model \\")
    print("    --train_data_file /path/to/data.json \\")
    print("    --enable_strategic_logging \\")
    print("    --strategic_logging_steps 100")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)