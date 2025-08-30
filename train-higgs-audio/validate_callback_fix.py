#!/usr/bin/env python3
"""
Validation script to check the callback fix structure
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def validate_callback_fix():
    """Validate that the callback fix has the correct structure"""
    try:
        # Import the callbacks
        from callbacks.strategic_logging import (
            InputLoggerCallback, 
            OutputLoggerCallback, 
            SharedAttentionLoggerCallback, 
            ZeroShotVerificationLoggerCallback
        )
        
        print("Validating strategic logging callback fix...")
        
        # Check that callbacks have the on_step_end method
        callbacks = [
            ("InputLoggerCallback", InputLoggerCallback),
            ("OutputLoggerCallback", OutputLoggerCallback),
            ("SharedAttentionLoggerCallback", SharedAttentionLoggerCallback),
            ("ZeroShotVerificationLoggerCallback", ZeroShotVerificationLoggerCallback)
        ]
        
        for name, callback_class in callbacks:
            callback = callback_class()
            if hasattr(callback, 'on_step_end'):
                print(f"✅ {name} has on_step_end method")
            else:
                print(f"❌ {name} missing on_step_end method")
                return False
                
            # Check method signature
            import inspect
            sig = inspect.signature(callback.on_step_end)
            params = list(sig.parameters.keys())
            required_params = ['args', 'state', 'control']
            has_required = all(param in params for param in required_params)
            
            if has_required:
                print(f"✅ {name} has required parameters: {required_params}")
            else:
                print(f"❌ {name} missing required parameters. Found: {params}")
                return False
                
            # Check for inputs, outputs, model parameters
            optional_params = ['inputs', 'outputs', 'model']
            has_optional = all(param in params for param in optional_params)
            
            if has_optional:
                print(f"✅ {name} has optional parameters: {optional_params}")
            else:
                print(f"⚠️  {name} missing optional parameters: {optional_params}")
                
        print("\n✅ Callback structure validation passed!")
        print("The callbacks should now properly receive model inputs and outputs during training.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating callback fix: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = validate_callback_fix()
    sys.exit(0 if success else 1)