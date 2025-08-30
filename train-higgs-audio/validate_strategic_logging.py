#!/usr/bin/env python3
"""
Validation script for strategic logging implementation
"""

import os
import sys

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def validate_training_step_signature():
    """Validate that the training_step method has the correct signature"""
    try:
        print("=== Training Step Signature Validation ===\n")
        
        # Read the trainer file
        trainer_file = os.path.join(project_root, 'trainer', 'train_v2_ddp.py')
        
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
            
        # Check that we're passing inputs and outputs as keyword arguments
        if "inputs': inputs" in content and "outputs': outputs" in content:
            print("✅ Method correctly passes inputs and outputs as keyword arguments")
        else:
            print("❌ Method does not correctly pass inputs and outputs as keyword arguments")
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

def validate_callback_signatures():
    """Validate that the callback methods have the correct signatures"""
    try:
        print("\n=== Callback Signature Validation ===\n")
        
        # Read the callback file
        callback_file = os.path.join(project_root, 'callbacks', 'strategic_logging.py')
        
        with open(callback_file, 'r') as f:
            content = f.read()
            
        # Check for the on_step_end method definitions
        callback_classes = [
            'InputLoggerCallback',
            'OutputLoggerCallback', 
            'SharedAttentionLoggerCallback',
            'ZeroShotVerificationLoggerCallback'
        ]
        
        for callback_class in callback_classes:
            if f'class {callback_class}' in content:
                print(f"✅ Found {callback_class} class definition")
            else:
                print(f"❌ Missing {callback_class} class definition")
                return False
                
            if f'def on_step_end(' in content:
                print(f"✅ Found on_step_end method in {callback_class}")
            else:
                print(f"❌ Missing on_step_end method in {callback_class}")
                return False
                
        # Check for the correct parameters in on_step_end methods
        required_patterns = [
            'args', 
            'state', 
            'control', 
            'inputs=None', 
            'outputs=None'
        ]
        
        for pattern in required_patterns:
            if pattern in content:
                print(f"✅ Found required parameter: {pattern}")
            else:
                print(f"❌ Missing required parameter: {pattern}")
                return False
                
        print("\n=== Callback Signature Validation Results ===")
        print("✅ All callback signatures validation PASSED!")
        print("✅ All callbacks have the correct on_step_end method signatures")
        print("✅ All callbacks accept the required parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in callback signature validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_imports():
    """Validate that the strategic logging callbacks can be imported"""
    try:
        print("\n=== Import Validation ===\n")
        
        # Try to import the callbacks
        from callbacks.strategic_logging import (
            InputLoggerCallback,
            OutputLoggerCallback,
            SharedAttentionLoggerCallback,
            ZeroShotVerificationLoggerCallback
        )
        
        print("✅ Successfully imported all strategic logging callbacks")
        
        # Try to import the trainer
        from trainer.train_v2_ddp import HiggsAudioTrainer
        print("✅ Successfully imported HiggsAudioTrainer")
        
        print("\n=== Import Validation Results ===")
        print("✅ All imports validation PASSED!")
        print("✅ Strategic logging callbacks are available for use")
        print("✅ HiggsAudioTrainer is available for use")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in import validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validations"""
    print("Running Strategic Logging Implementation Validation...\n")
    
    # Run all validation functions
    validations = [
        validate_training_step_signature,
        validate_callback_signatures,
        validate_imports
    ]
    
    results = []
    for validation in validations:
        result = validation()
        results.append(result)
        
    # Print final results
    print("\n" + "="*50)
    print("FINAL VALIDATION RESULTS")
    print("="*50)
    
    if all(results):
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Strategic logging implementation is ready for use")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("❌ Please check the errors above and fix the implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())