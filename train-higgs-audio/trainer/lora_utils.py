#!/usr/bin/env python3
"""
Utility functions for LoRA adapter handling in Higgs Audio training
"""

import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


def find_lora_model(model: Any) -> Optional[Any]:
    """
    Robustly find the PEFT model component within the wrapped model
    
    Args:
        model: The model wrapper or model component
        
    Returns:
        The PEFT model component if found, None otherwise
    """
    # Check if it's already a PeftModel
    if hasattr(model, 'save_pretrained') and hasattr(model, 'peft_config'):
        logger.debug("Model is already a PeftModel")
        return model
    
    # Check common nested structures
    if hasattr(model, 'model'):
        if hasattr(model.model, 'save_pretrained') and hasattr(model.model, 'peft_config'):
            logger.debug("Found PeftModel in model.model")
            return model.model
        if hasattr(model.model, 'text_model') and hasattr(model.model.text_model, 'save_pretrained'):
            logger.debug("Found PeftModel in model.model.text_model")
            return model.model.text_model
    
    # For DDP wrapped models
    if hasattr(model, 'module'):
        module = model.module
        if hasattr(module, 'save_pretrained') and hasattr(module, 'peft_config'):
            logger.debug("Found PeftModel in model.module")
            return module
        if hasattr(module, 'model'):
            if hasattr(module.model, 'save_pretrained') and hasattr(module.model, 'peft_config'):
                logger.debug("Found PeftModel in model.module.model")
                return module.model
            if hasattr(module.model, 'text_model') and hasattr(module.model.text_model, 'save_pretrained'):
                logger.debug("Found PeftModel in model.module.model.text_model")
                return module.model.text_model
    
    logger.warning("Could not find LoRA model component for saving")
    return None


def save_lora_adapters(trainer: Any, output_dir: str, use_lora: bool) -> None:
    """
    Save LoRA adapters separately from model checkpoints
    
    Args:
        trainer: The trainer instance
        output_dir: The output directory for saving
        use_lora: Whether LoRA is enabled
    """
    if not use_lora:
        return
    
    logger.info("Attempting to save LoRA adapters...")
    
    # Find the correct model component
    lora_model = find_lora_model(trainer.model)
    
    if lora_model is None:
        logger.error("Could not find LoRA model component for saving")
        return
    
    # Create LoRA adapters directory
    lora_output_dir = os.path.join(output_dir, "lora_adapters")
    os.makedirs(lora_output_dir, exist_ok=True)
    logger.info(f"Created LoRA adapters directory: {lora_output_dir}")
    
    try:
        # Save the LoRA adapters
        lora_model.save_pretrained(lora_output_dir)
        logger.info(f"LoRA adapters successfully saved to {lora_output_dir}")
        
        # Log contents of the directory for verification
        if os.path.exists(lora_output_dir):
            contents = os.listdir(lora_output_dir)
            logger.info(f"Contents of lora_adapters directory: {contents}")
        else:
            logger.warning(f"LoRA adapters directory does not exist after save attempt: {lora_output_dir}")
            
    except Exception as e:
        logger.error(f"Failed to save LoRA adapters: {e}")
        logger.exception("Exception details:")