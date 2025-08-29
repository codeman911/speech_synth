#!/usr/bin/env python3
"""
Utility script to fix LoRA adapter saving issues in Higgs Audio training outputs.
This script can be run after training to ensure LoRA adapters are properly saved.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_lora_model(model):
    """
    Robustly find the PEFT model component within the wrapped model
    """
    # Check if it's already a PeftModel
    if hasattr(model, 'save_pretrained') and hasattr(model, 'peft_config'):
        return model
    
    # Check common nested structures
    if hasattr(model, 'model'):
        if hasattr(model.model, 'save_pretrained') and hasattr(model.model, 'peft_config'):
            return model.model
        if hasattr(model.model, 'text_model') and hasattr(model.model.text_model, 'save_pretrained'):
            return model.model.text_model
    
    # For DDP wrapped models
    if hasattr(model, 'module'):
        module = model.module
        if hasattr(module, 'save_pretrained') and hasattr(module, 'peft_config'):
            return module
        if hasattr(module, 'model'):
            if hasattr(module.model, 'save_pretrained') and hasattr(module.model, 'peft_config'):
                return module.model
            if hasattr(module.model, 'text_model') and hasattr(module.model.text_model, 'save_pretrained'):
                return module.model.text_model
    
    return None


def save_lora_adapters_from_checkpoint(checkpoint_path, output_dir=None):
    """
    Extract and save LoRA adapters from a trained model checkpoint
    
    Args:
        checkpoint_path (str): Path to the trained model checkpoint
        output_dir (str): Output directory for LoRA adapters (default: checkpoint_path/lora_adapters)
    """
    try:
        import torch
        from peft import PeftModel
    except ImportError:
        logger.error("Required packages not installed. Please install torch and peft.")
        return False
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
        return False
    
    if output_dir is None:
        output_dir = os.path.join(checkpoint_path, "lora_adapters")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    try:
        # Try to load the model
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Check if it's a PEFT model directly
        if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
            logger.info("Checkpoint is already a LoRA adapter directory")
            # Just copy the files
            import shutil
            for file in os.listdir(checkpoint_path):
                if file.startswith("adapter_"):
                    src = os.path.join(checkpoint_path, file)
                    dst = os.path.join(output_dir, file)
                    shutil.copy2(src, dst)
                    logger.info(f"Copied {file} to {output_dir}")
            return True
        
        # Try to load as a full model and extract LoRA adapters
        # We need to import the model classes
        try:
            from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
            config = HiggsAudioConfig.from_pretrained(checkpoint_path)
            model = HiggsAudioModel.from_pretrained(checkpoint_path, config=config)
        except Exception as e:
            logger.warning(f"Could not load HiggsAudioModel: {e}")
            # Try loading with AutoModel
            from transformers import AutoModel
            model = AutoModel.from_pretrained(checkpoint_path)
        
        # Find the LoRA model component
        lora_model = find_lora_model(model)
        
        if lora_model is None:
            logger.error("Could not find LoRA model component in the checkpoint")
            return False
        
        # Save the LoRA adapters
        logger.info("Saving LoRA adapters...")
        lora_model.save_pretrained(output_dir)
        logger.info(f"LoRA adapters successfully saved to {output_dir}")
        
        # Verify the saved files
        if os.path.exists(output_dir):
            contents = os.listdir(output_dir)
            logger.info(f"Contents of lora_adapters directory: {contents}")
            
            required_files = ["adapter_config.json"]
            if any(f.startswith("adapter_model") and f.endswith((".bin", ".safetensors")) for f in contents):
                required_files.append("adapter_model file")
            
            missing_files = []
            for required in required_files:
                if required == "adapter_config.json" and "adapter_config.json" not in contents:
                    missing_files.append(required)
                elif required == "adapter_model file":
                    if not any(f.startswith("adapter_model") and f.endswith((".bin", ".safetensors")) for f in contents):
                        missing_files.append(required)
            
            if missing_files:
                logger.warning(f"Missing required files: {missing_files}")
                return False
            else:
                logger.info("All required LoRA adapter files are present")
                return True
        else:
            logger.error(f"Failed to create LoRA adapters directory: {output_dir}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to extract LoRA adapters: {e}")
        logger.exception("Exception details:")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fix LoRA adapter saving issues in Higgs Audio training outputs")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the trained model checkpoint directory")
    parser.add_argument("--output_dir", type=str, 
                        help="Output directory for LoRA adapters (default: checkpoint_path/lora_adapters)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting LoRA adapter extraction...")
    logger.info(f"Checkpoint path: {args.checkpoint_path}")
    logger.info(f"Output directory: {args.output_dir or 'default'}")
    
    success = save_lora_adapters_from_checkpoint(args.checkpoint_path, args.output_dir)
    
    if success:
        logger.info("LoRA adapter extraction completed successfully!")
        return 0
    else:
        logger.error("LoRA adapter extraction failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())