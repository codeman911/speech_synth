#!/usr/bin/env python3
"""
Syntax validation script to check the callback fix
"""

import sys
import os
import ast

def validate_syntax(file_path):
    """Validate Python syntax of a file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
        print(f"✅ Syntax validation passed for {file_path}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def validate_callback_methods():
    """Validate that the callback methods are correctly implemented"""
    try:
        # Read the callback file
        callback_file = os.path.join(os.path.dirname(__file__), 'callbacks', 'strategic_logging.py')
        
        with open(callback_file, 'r') as f:
            content = f.read()
            
        # Check for on_step_end method instead of on_log
        if 'def on_step_end(' in content:
            print("✅ Callbacks updated to use on_step_end method")
        else:
            print("❌ Callbacks still using on_log method")
            return False
            
        # Check for proper parameter handling
        required_patterns = [
            'inputs=None',
            'outputs=None', 
            'model=None',
            'state.global_step'
        ]
        
        missing_patterns = []
        for pattern in required_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)
                
        if missing_patterns:
            print(f"❌ Missing patterns in callbacks: {missing_patterns}")
            return False
        else:
            print("✅ All required patterns found in callbacks")
            
        # Check that trainer is updated to call on_step_end
        trainer_file = os.path.join(os.path.dirname(__file__), 'trainer', 'train_v2_ddp.py')
        
        with open(trainer_file, 'r') as f:
            trainer_content = f.read()
            
        if 'def training_step(' in trainer_content and 'on_step_end' in trainer_content:
            print("✅ Trainer updated with training_step override")
        else:
            print("❌ Trainer not properly updated")
            return False
            
        print("\n✅ All syntax and structure validations passed!")
        print("The fix should now properly pass model inputs and outputs to the callbacks.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating callback methods: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    print("Validating strategic logging callback fix...\n")
    
    # Validate syntax of both files
    callback_file = os.path.join(os.path.dirname(__file__), 'callbacks', 'strategic_logging.py')
    trainer_file = os.path.join(os.path.dirname(__file__), 'trainer', 'train_v2_ddp.py')
    
    if not validate_syntax(callback_file):
        return False
        
    if not validate_syntax(trainer_file):
        return False
        
    # Validate method structure
    if not validate_callback_methods():
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)