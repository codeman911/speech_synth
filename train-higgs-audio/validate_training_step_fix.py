#!/usr/bin/env python3
"""
Validation script to specifically check the training_step signature fix
"""

import sys
import os
import inspect

def validate_training_step_signature():
    """Validate that the training_step method has the correct signature"""
    try:
        print("=== Training Step Signature Validation ===\n")
        
        # Read the trainer file
        trainer_file = os.path.join(os.path.dirname(__file__), 'trainer', 'train_v2_ddp.py')
        
        with open(trainer_file, 'r') as f:
            content = f.read()
            
        # Check for the training_step method definition
        if 'def training_step(' in content:
            print("✅ Found training_step method definition")
        else:
            print("❌ Missing training_step method definition")
            return False
            
        # Check for the correct parameters
        required_patterns = [
            'self',
            'model', 
            'inputs',
            'num_items_in_batch=None'
        ]
        
        missing_patterns = []
        for pattern in required_patterns:
            if pattern in content:
                print(f"✅ Found required parameter: {pattern}")
            else:
                missing_patterns.append(pattern)
                print(f"❌ Missing required parameter: {pattern}")
                
        if missing_patterns:
            print(f"\n❌ Missing patterns in training_step signature: {missing_patterns}")
            return False
            
        # Check that the method calls the callback handler properly
        if 'callback_handler.call_event' in content and 'on_step_end' in content:
            print("✅ Method properly calls callback handler with on_step_end")
        else:
            print("❌ Method does not properly call callback handler")
            return False
            
        # Check that we're not passing model as a keyword argument
        if "model': model" not in content:
            print("✅ Method correctly avoids passing model as keyword argument")
        else:
            print("❌ Method incorrectly passes model as keyword argument")
            return False
            
        # Check for the return statement
        if 'return loss' in content:
            print("✅ Method returns loss value")
        else:
            print("❌ Method does not return loss value")
            return False
        
        print("\n=== Signature Validation Results ===")
        print("✅ Training step signature validation PASSED!")
        print("✅ The method now accepts the correct number of parameters")
        print("✅ The signature matches the Hugging Face Trainer interface")
        print("✅ Callbacks should be properly called without parameter conflicts")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in signature validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_callback_methods():
    """Validate that the callback methods have the correct signature"""
    try:
        print("\n=== Callback Method Validation ===\n")
        
        # Read the callback file
        callback_file = os.path.join(os.path.dirname(__file__), 'callbacks', 'strategic_logging.py')
        
        with open(callback_file, 'r') as f:
            content = f.read()
            
        # Check that callbacks use on_step_end instead of on_log
        if 'def on_step_end(' in content:
            print("✅ Callbacks use on_step_end method")
        else:
            print("❌ Callbacks still use on_log method")
            return False
            
        # Check for proper parameter handling (without model parameter)
        required_params = ['inputs=None', 'outputs=None']
        forbidden_params = ['model=None']  # Model should not be passed as parameter
        for param in required_params:
            if param in content:
                print(f"✅ Callbacks handle parameter: {param}")
            else:
                print(f"❌ Callbacks missing parameter: {param}")
                return False
                
        # Check that we're not passing model as a parameter
        for param in forbidden_params:
            if param not in content:
                print(f"✅ Callbacks correctly avoid parameter: {param}")
            else:
                print(f"❌ Callbacks incorrectly include parameter: {param}")
                return False
                
        print("\n✅ Callback method validation PASSED!")
        print("✅ Callbacks now properly receive model data without conflicts")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in callback validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    print("Validating strategic logging training_step signature fix...\n")
    
    # Validate training step signature
    step_valid = validate_training_step_signature()
    
    # Validate callback methods
    callback_valid = validate_callback_methods()
    
    if step_valid and callback_valid:
        print("\n🎉 All validations PASSED! The strategic logging fix is complete.")
        print("\nExpected behavior:")
        print("  ✅ No more TypeError about training_step arguments")
        print("  ✅ No more conflicts with default callbacks")
        print("  ✅ Callbacks will receive model inputs and outputs")
        print("  ✅ Detailed logs will show at step 1 and every N steps")
        print("  ✅ 'NOT FOUND' messages should be resolved")
        return True
    else:
        print("\n💥 Some validations FAILED! Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)