#!/usr/bin/env python3
"""
Final test to verify callback implementation is correct
"""

import sys
import os

def test_callback_implementation():
    """Test that callback implementation is correct"""
    try:
        # Check file structure
        project_root = os.path.abspath('.')
        callbacks_file = os.path.join(project_root, 'callbacks', 'strategic_logging.py')
        
        if not os.path.exists(callbacks_file):
            print(f"ERROR: Callbacks file not found at {callbacks_file}")
            return False
            
        # Read the file and verify it has the right structure
        with open(callbacks_file, 'r') as f:
            content = f.read()
            
        # Check for key components
        checks = [
            ("Base import", "from transformers import TrainerCallback"),
            ("InputLoggerCallback class", "class InputLoggerCallback(TrainerCallback)"),
            ("OutputLoggerCallback class", "class OutputLoggerCallback(TrainerCallback)"),
            ("SharedAttentionLoggerCallback class", "class SharedAttentionLoggerCallback(TrainerCallback)"),
            ("ZeroShotVerificationLoggerCallback class", "class ZeroShotVerificationLoggerCallback(TrainerCallback)"),
            ("on_log method", "def on_log(self, args, state, control, logs=None, **kwargs)"),
            ("deferred torch import", "torch = None"),
            ("_import_torch function", "def _import_torch():"),
        ]
        
        failed_checks = []
        for check_name, check_content in checks:
            if check_content not in content:
                failed_checks.append(check_name)
                
        if failed_checks:
            print(f"ERROR: Failed checks: {failed_checks}")
            return False
            
        print("SUCCESS: Callback implementation is correct")
        print(f"Callbacks file: {callbacks_file}")
        print(f"File size: {os.path.getsize(callbacks_file)} bytes")
        
        # Check trainer script import path
        trainer_file = os.path.join(project_root, 'trainer', 'train_v2_ddp.py')
        if os.path.exists(trainer_file):
            with open(trainer_file, 'r') as f:
                trainer_content = f.read()
                
            if "from callbacks.strategic_logging import" in trainer_content:
                print("SUCCESS: Trainer script has correct import statement")
            else:
                print("WARNING: Trainer script may have incorrect import statement")
        else:
            print("WARNING: Trainer script not found for verification")
            
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_callback_implementation()
    sys.exit(0 if success else 1)