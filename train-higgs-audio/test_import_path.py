#!/usr/bin/env python3
"""
Test script to verify import path without requiring dependencies
"""

import sys
import os

def test_import_path():
    """Test import path by checking sys.path and file structure"""
    try:
        # Current working directory should be the project root
        project_root = os.path.abspath('.')
        print(f"Project root: {project_root}")
        
        # Check if callbacks directory exists
        callbacks_dir = os.path.join(project_root, 'callbacks')
        if not os.path.exists(callbacks_dir):
            print(f"ERROR: Callbacks directory not found at {callbacks_dir}")
            return False
            
        # Check if strategic_logging.py exists
        strategic_logging_file = os.path.join(callbacks_dir, 'strategic_logging.py')
        if not os.path.exists(strategic_logging_file):
            print(f"ERROR: strategic_logging.py not found at {strategic_logging_file}")
            return False
            
        # Check if __init__.py exists in callbacks directory
        init_file = os.path.join(callbacks_dir, '__init__.py')
        if not os.path.exists(init_file):
            print(f"ERROR: __init__.py not found at {init_file}")
            return False
            
        # Add project root to sys.path if not already there
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            print(f"Added {project_root} to sys.path")
            
        print("SUCCESS: Import path structure is correct")
        print(f"sys.path[0]: {sys.path[0]}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_import_path()
    sys.exit(0 if success else 1)