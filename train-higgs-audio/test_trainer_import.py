#!/usr/bin/env python3
"""
Test script to simulate trainer script importing callbacks
"""

import sys
import os

def test_trainer_import():
    """Test trainer import simulation"""
    try:
        # Simulate the trainer script environment
        project_root = os.path.abspath('.')
        trainer_dir = os.path.join(project_root, 'trainer')
        
        print(f"Project root: {project_root}")
        print(f"Trainer directory: {trainer_dir}")
        
        # Add project root to sys.path (this is what we added to the trainer script)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            print(f"Added {project_root} to sys.path")
            
        # Try to import the callbacks (this is what the trainer script does)
        print("Attempting to import callbacks...")
        from callbacks.strategic_logging import (
            InputLoggerCallback, 
            OutputLoggerCallback, 
            SharedAttentionLoggerCallback, 
            ZeroShotVerificationLoggerCallback
        )
        
        print("SUCCESS: Callbacks imported successfully in trainer context")
        print(f"InputLoggerCallback: {InputLoggerCallback}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to import callbacks: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trainer_import()
    sys.exit(0 if success else 1)