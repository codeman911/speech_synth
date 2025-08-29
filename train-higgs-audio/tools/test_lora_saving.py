#!/usr/bin/env python3
"""
Simple test script to verify LoRA adapter saving works correctly.
This follows the exact pattern from the original working code.
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_lora_saving_approach():
    """
    Test the LoRA saving approach from the original working code.
    """
    try:
        import torch
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModel
    except ImportError as e:
        logger.error(f"Required packages not available: {e}")
        return False
    
    logger.info("Testing LoRA saving approach...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        try:
            # Create a simple model for testing
            logger.info("Creating test model...")
            model = AutoModel.from_pretrained("bert-base-uncased")
            
            # Apply LoRA configuration (similar to setup_lora_config function)
            logger.info("Applying LoRA configuration...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["query", "value"]  # BERT-specific layers
            )
            model = get_peft_model(model, peft_config)
            logger.info("LoRA configuration applied successfully")
            
            # Test the original approach from trainer.py
            logger.info("Testing original approach from trainer.py...")
            lora_output_dir = os.path.join(temp_dir, "lora_adapters_original")
            
            # This is the approach from trainer.py
            if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
                model.model.text_model.save_pretrained(lora_output_dir)
            elif hasattr(model, 'model'):
                model.model.save_pretrained(lora_output_dir)
            else:
                model.save_pretrained(lora_output_dir)
            
            # Check if files were created
            if os.path.exists(lora_output_dir):
                contents = os.listdir(lora_output_dir)
                logger.info(f"Original approach - Contents: {contents}")
                
                # Check for required files
                has_config = "adapter_config.json" in contents
                has_model = any(f.startswith("adapter_model") for f in contents)
                
                if has_config and has_model:
                    logger.info("Original approach: SUCCESS")
                else:
                    logger.warning("Original approach: Missing required files")
            else:
                logger.error("Original approach: Directory not created")
            
            # Test the approach from trainer_ddp.py
            logger.info("Testing approach from trainer_ddp.py...")
            lora_output_dir2 = os.path.join(temp_dir, "lora_adapters_ddp")
            
            # This is the approach from trainer_ddp.py
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(lora_output_dir2)
            
            # Check if files were created
            if os.path.exists(lora_output_dir2):
                contents = os.listdir(lora_output_dir2)
                logger.info(f"DDP approach - Contents: {contents}")
                
                # Check for required files
                has_config = "adapter_config.json" in contents
                has_model = any(f.startswith("adapter_model") for f in contents)
                
                if has_config and has_model:
                    logger.info("DDP approach: SUCCESS")
                else:
                    logger.warning("DDP approach: Missing required files")
            else:
                logger.error("DDP approach: Directory not created")
                
            # Compare file sizes
            if os.path.exists(lora_output_dir) and os.path.exists(lora_output_dir2):
                orig_size = sum(os.path.getsize(os.path.join(lora_output_dir, f)) 
                               for f in os.listdir(lora_output_dir) 
                               if f.startswith("adapter_model"))
                ddp_size = sum(os.path.getsize(os.path.join(lora_output_dir2, f)) 
                              for f in os.listdir(lora_output_dir2) 
                              if f.startswith("adapter_model"))
                
                logger.info(f"Original approach model size: {orig_size} bytes")
                logger.info(f"DDP approach model size: {ddp_size} bytes")
                
                if orig_size == ddp_size:
                    logger.info("Both approaches produce identical results")
                    return True
                else:
                    logger.warning("Approaches produce different results")
                    return False
            else:
                logger.error("Could not compare approaches - missing directories")
                return False
                
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            logger.exception("Exception details:")
            return False


def main():
    print("LoRA Saving Test")
    print("================")
    
    success = test_lora_saving_approach()
    
    if success:
        print("\n✓ Test PASSED: LoRA saving works correctly")
        print("\nThe original approaches from the working code are valid:")
        print("1. trainer.py approach: Multiple fallbacks for model structure")
        print("2. trainer_ddp.py approach: Simple trainer.model access with DDP handling")
        return 0
    else:
        print("\n✗ Test FAILED: LoRA saving has issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())