#!/usr/bin/env python3
"""
Utility script to verify that LoRA adapters were saved correctly.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_lora_adapters(checkpoint_path):
    """
    Verify that LoRA adapters were saved correctly in the given checkpoint path.
    
    Args:
        checkpoint_path (str): Path to the checkpoint directory or lora_adapters subdirectory
        
    Returns:
        bool: True if LoRA adapters are valid, False otherwise
    """
    # Check if the path is already pointing to lora_adapters directory
    if os.path.basename(checkpoint_path) == "lora_adapters":
        lora_dir = checkpoint_path
    else:
        # Look for lora_adapters subdirectory
        lora_dir = os.path.join(checkpoint_path, "lora_adapters")
    
    logger.info(f"Checking LoRA adapters in: {lora_dir}")
    
    # Check if directory exists
    if not os.path.exists(lora_dir):
        logger.error(f"LoRA adapters directory not found: {lora_dir}")
        # Also check if this is already a LoRA adapter directory
        if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
            logger.info("This appears to already be a LoRA adapter directory")
            lora_dir = checkpoint_path
        else:
            return False
    
    # List contents
    try:
        contents = os.listdir(lora_dir)
        logger.info(f"Contents of directory: {contents}")
    except Exception as e:
        logger.error(f"Failed to list directory contents: {e}")
        return False
    
    # Check for required files
    required_files = ["adapter_config.json"]
    adapter_model_files = [f for f in contents if f.startswith("adapter_model") and f.endswith((".bin", ".safetensors"))]
    
    missing_files = []
    
    # Check for adapter_config.json
    if "adapter_config.json" not in contents:
        missing_files.append("adapter_config.json")
    else:
        # Try to load and validate adapter_config.json
        try:
            config_path = os.path.join(lora_dir, "adapter_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Adapter config loaded successfully")
            logger.info(f"Adapter type: {config.get('peft_type', 'Unknown')}")
            logger.info(f"Base model type: {config.get('base_model_name_or_path', 'Unknown')}")
            
            # Check for required config fields
            required_config_fields = ['peft_type', 'base_model_name_or_path', 'task_type']
            missing_config_fields = [field for field in required_config_fields if field not in config]
            if missing_config_fields:
                logger.warning(f"Missing fields in adapter_config.json: {missing_config_fields}")
        except Exception as e:
            logger.error(f"Failed to load adapter_config.json: {e}")
            missing_files.append("adapter_config.json")
    
    # Check for adapter_model file
    if not adapter_model_files:
        missing_files.append("adapter_model file (*.bin or *.safetensors)")
    else:
        logger.info(f"Found adapter model files: {adapter_model_files}")
        # Check file sizes
        for adapter_file in adapter_model_files:
            file_path = os.path.join(lora_dir, adapter_file)
            size = os.path.getsize(file_path)
            logger.info(f"  {adapter_file}: {size} bytes ({size / 1024 / 1024:.2f} MB)")
    
    # Report results
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    else:
        logger.info("All required LoRA adapter files are present and valid")
        return True


def find_lora_directories(base_path):
    """
    Find all lora_adapters directories in the given base path.
    
    Args:
        base_path (str): Base path to search
        
    Returns:
        list: List of paths to lora_adapters directories
    """
    lora_dirs = []
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(base_path):
        # Check if this directory is named "lora_adapters"
        if os.path.basename(root) == "lora_adapters":
            # Check if it contains the required files
            adapter_config = os.path.join(root, "adapter_config.json")
            adapter_model_bin = os.path.join(root, "adapter_model.bin")
            adapter_model_safetensors = os.path.join(root, "adapter_model.safetensors")
            
            if os.path.exists(adapter_config) and (os.path.exists(adapter_model_bin) or os.path.exists(adapter_model_safetensors)):
                lora_dirs.append(root)
                logger.info(f"Found valid LoRA adapters directory: {root}")
    
    return lora_dirs


def main():
    parser = argparse.ArgumentParser(description="Verify LoRA adapters were saved correctly")
    parser.add_argument("--path", type=str, required=True, 
                        help="Path to checkpoint directory or lora_adapters directory")
    parser.add_argument("--find-all", action="store_true",
                        help="Search recursively for all lora_adapters directories")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.find_all:
        logger.info(f"Searching for all lora_adapters directories in: {args.path}")
        lora_dirs = find_lora_directories(args.path)
        
        if lora_dirs:
            logger.info(f"Found {len(lora_dirs)} lora_adapters directories:")
            for i, dir_path in enumerate(lora_dirs, 1):
                logger.info(f"{i}. {dir_path}")
        else:
            logger.info("No valid lora_adapters directories found.")
        return
    
    # Verify specific path
    success = verify_lora_adapters(args.path)
    
    if success:
        logger.info("Verification PASSED: LoRA adapters are valid")
        return 0
    else:
        logger.error("Verification FAILED: LoRA adapters are not valid")
        return 1


if __name__ == "__main__":
    sys.exit(main())