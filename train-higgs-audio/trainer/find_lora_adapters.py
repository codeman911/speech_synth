#!/usr/bin/env python3
"""
Helper script to find LoRA adapters directory
"""

import os
import sys
import argparse
from pathlib import Path

def find_lora_adapters(base_path):
    """Find lora_adapters directories in the given path"""
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
                print(f"Found valid LoRA adapters directory: {root}")
                print(f"  - adapter_config.json: {'✓' if os.path.exists(adapter_config) else '✗'}")
                print(f"  - adapter_model.bin: {'✓' if os.path.exists(adapter_model_bin) else '✗'}")
                print(f"  - adapter_model.safetensors: {'✓' if os.path.exists(adapter_model_safetensors) else '✗'}")
                print()
    
    return lora_dirs

def main():
    parser = argparse.ArgumentParser(description="Find LoRA adapters directories")
    parser.add_argument("--path", type=str, default=".", help="Base path to search (default: current directory)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    
    args = parser.parse_args()
    
    print(f"Searching for LoRA adapters directories in: {os.path.abspath(args.path)}")
    print("=" * 60)
    
    lora_dirs = find_lora_adapters(args.path)
    
    if lora_dirs:
        print(f"Found {len(lora_dirs)} LoRA adapters directory/directories:")
        for i, dir_path in enumerate(lora_dirs, 1):
            print(f"{i}. {dir_path}")
    else:
        print("No valid LoRA adapters directories found.")
        print("\nTips:")
        print("1. Make sure you have run training with --use_lora flag")
        print("2. LoRA adapters are saved in a separate 'lora_adapters' directory")
        print("3. Checkpoint directories do NOT contain LoRA adapters")
        print("4. Look for directories named 'lora_adapters' in your training output folders")

if __name__ == "__main__":
    main()