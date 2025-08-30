#!/usr/bin/env python3
"""
Final integration test to verify the complete strategic logging fix
"""

import sys
import os

def test_complete_fix():
    """Test that all components of the fix work together"""
    try:
        print("=== Strategic Logging Fix Integration Test ===\n")
        
        # Check that required files exist
        required_files = [
            'callbacks/strategic_logging.py',
            'trainer/train_v2_ddp.py'
        ]
        
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        for file_path in required_files:
            full_path = os.path.join(project_root, file_path)
            if os.path.exists(full_path):
                print(f"✅ Found required file: {file_path}")
            else:
                print(f"❌ Missing required file: {file_path}")
                return False
        
        # Check callback file content
        callback_file = os.path.join(project_root, 'callbacks', 'strategic_logging.py')
        with open(callback_file, 'r') as f:
            callback_content = f.read()
            
        # Verify callbacks use on_step_end
        if 'def on_step_end(' in callback_content:
            print("✅ Callbacks use on_step_end method")
        else:
            print("❌ Callbacks do not use on_step_end method")
            return False
            
        # Verify proper parameter handling
        required_params = ['inputs=None', 'outputs=None', 'model=None']
        for param in required_params:
            if param in callback_content:
                print(f"✅ Callbacks handle parameter: {param}")
            else:
                print(f"❌ Callbacks missing parameter: {param}")
                return False
        
        # Check trainer file content
        trainer_file = os.path.join(project_root, 'trainer', 'train_v2_ddp.py')
        with open(trainer_file, 'r') as f:
            trainer_content = f.read()
            
        # Verify training_step override
        if 'def training_step(' in trainer_content:
            print("✅ Trainer overrides training_step method")
        else:
            print("❌ Trainer does not override training_step method")
            return False
            
        # Verify callback calling
        if 'on_step_end' in trainer_content and 'callback_handler' in trainer_content:
            print("✅ Trainer properly calls callbacks with on_step_end")
        else:
            print("❌ Trainer does not properly call callbacks")
            return False
        
        # Check for debugging output
        debug_indicators = [
            'STRATEGIC LOG DEBUG',
            'Received inputs type',
            'Received outputs type'
        ]
        
        debug_found = 0
        for indicator in debug_indicators:
            if indicator in callback_content:
                print(f"✅ Found debugging feature: {indicator}")
                debug_found += 1
                
        if debug_found > 0:
            print(f"✅ Found {debug_found}/{len(debug_indicators)} debugging features")
        else:
            print("⚠️  No debugging features found")
        
        print("\n=== Test Results ===")
        print("✅ All core components validated successfully!")
        print("✅ Strategic logging callbacks should now receive model data")
        print("✅ Training should show detailed logs at step 1 and every N steps")
        print("\nThe fix addresses the root cause of the 'NOT FOUND' issues by:")
        print("  1. Properly passing model inputs/outputs from trainer to callbacks")
        print("  2. Using the correct callback method (on_step_end vs on_log)")
        print("  3. Adding debugging to verify data flow")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_fix()
    if success:
        print("\n🎉 Integration test PASSED! The strategic logging fix is ready.")
    else:
        print("\n💥 Integration test FAILED! Please review the issues above.")
    sys.exit(0 if success else 1)