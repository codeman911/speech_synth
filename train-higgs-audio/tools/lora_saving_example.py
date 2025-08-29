#!/usr/bin/env python3
"""
Example script demonstrating the correct LoRA adapter saving approach
that follows the original code patterns.
"""

import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_lora_adapters_correctly(trainer, output_dir, use_lora):
    """
    Save LoRA adapters correctly following the original code patterns.
    
    This function demonstrates the approach that works reliably:
    1. Use trainer.model instead of the original model
    2. Handle DDP wrapped models properly
    3. Include error handling
    4. Verify the saved files
    
    Args:
        trainer: The trainer instance from HuggingFace
        output_dir (str): The base output directory
        use_lora (bool): Whether LoRA training was enabled
    """
    if not use_lora:
        return
    
    try:
        # Create LoRA adapters directory
        lora_output_dir = os.path.join(output_dir, "lora_adapters")
        os.makedirs(lora_output_dir, exist_ok=True)
        logger.info(f"Created LoRA adapters directory: {lora_output_dir}")
        
        # Use trainer.model instead of original model for LoRA saving
        # This is the key fix - always use the trainer's model
        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        
        # Save the LoRA adapters
        model_to_save.save_pretrained(lora_output_dir)
        logger.info(f"LoRA adapters saved to {lora_output_dir}")
        
        # Verify saved files
        if os.path.exists(lora_output_dir):
            contents = os.listdir(lora_output_dir)
            logger.info(f"Contents of lora_adapters directory: {contents}")
            
            # Check for required files
            required_files = ["adapter_config.json"]
            adapter_model_files = [f for f in contents if f.startswith("adapter_model") and f.endswith((".bin", ".safetensors"))]
            
            if "adapter_config.json" in contents and len(adapter_model_files) > 0:
                logger.info("LoRA adapters saved successfully with all required files")
                logger.info(f"Adapter model files: {adapter_model_files}")
            else:
                logger.warning("LoRA adapters directory may be missing required files")
                logger.warning("Required: adapter_config.json and adapter_model file")
                logger.warning(f"Found: {contents}")
        else:
            logger.error(f"LoRA adapters directory was not created: {lora_output_dir}")
            
    except Exception as e:
        logger.error(f"Failed to save LoRA adapters: {e}")
        logger.exception("Exception details:")
        raise


# Example usage in a training script:
"""
# In your training script, after trainer.train(), replace the existing LoRA saving code with:

if trainer.is_world_process_zero():  # For DDP training
    trainer.save_model()
    logger.info(f"Model saved to {args.output_dir}")
    
    # Save LoRA adapters separately
    save_lora_adapters_correctly(trainer, args.output_dir, args.use_lora)

# For single-GPU training, you can skip the is_world_process_zero() check:
    
trainer.save_model()
logger.info(f"Model saved to {args.output_dir}")

# Save LoRA adapters separately
save_lora_adapters_correctly(trainer, args.output_dir, args.use_lora)
"""


def main():
    print("LoRA Adapter Saving Example")
    print("==========================")
    print("This script demonstrates the correct approach to saving LoRA adapters")
    print("in Higgs Audio training scripts.")
    print("\nKey points:")
    print("1. Always use trainer.model instead of the original model")
    print("2. Handle DDP wrapped models with hasattr(trainer.model, 'module')")
    print("3. Include proper error handling")
    print("4. Verify that required files are saved")
    print("\nTo use this in your training script:")
    print("1. Copy the save_lora_adapters_correctly function to your script")
    print("2. Replace your existing LoRA saving code with a call to this function")


if __name__ == "__main__":
    main()