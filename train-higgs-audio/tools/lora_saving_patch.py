#!/usr/bin/env python3
"""
Patch script to fix LoRA adapter saving in Higgs Audio training scripts.
This script follows the original approach but with enhanced error handling.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def patch_trainer_script(script_path):
    """
    Patch a trainer script to fix LoRA adapter saving.
    
    Args:
        script_path (str): Path to the trainer script to patch
    """
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
    
    # Read the script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Define the original LoRA saving patterns we want to fix
    original_patterns = [
        # Pattern from trainer.py
        ('        if hasattr(model, \'model\') and hasattr(model.model, \'text_model\'):\n            model.model.text_model.save_pretrained(lora_output_dir)\n        elif hasattr(model, \'model\'):\n            model.model.save_pretrained(lora_output_dir)\n        else:\n            model.save_pretrained(lora_output_dir)',
         '        # Use trainer.model instead of original model for LoRA saving\n        model_to_save = trainer.model.module if hasattr(trainer.model, \'module\') else trainer.model\n        model_to_save.save_pretrained(lora_output_dir)'),
        
        # Pattern from trainer_ddp.py
        ('        if args.use_lora:\n            lora_output_dir = os.path.join(args.output_dir, "lora_adapters")\n            model_to_save = trainer.model.module if hasattr(trainer.model, \'module\') else trainer.model\n            model_to_save.save_pretrained(lora_output_dir)\n            logger.info(f"LoRA adapters saved to {lora_output_dir}")',
         '        if args.use_lora:\n            lora_output_dir = os.path.join(args.output_dir, "lora_adapters")\n            # Enhanced LoRA saving with error handling\n            try:\n                model_to_save = trainer.model.module if hasattr(trainer.model, \'module\') else trainer.model\n                model_to_save.save_pretrained(lora_output_dir)\n                logger.info(f"LoRA adapters saved to {lora_output_dir}")\n                # Verify saved files\n                if os.path.exists(lora_output_dir):\n                    contents = os.listdir(lora_output_dir)\n                    logger.info(f"Contents of lora_adapters: {contents}")\n                else:\n                    logger.warning(f"LoRA adapters directory was not created: {lora_output_dir}")\n            except Exception as e:\n                logger.error(f"Failed to save LoRA adapters: {e}")\n                logger.exception("Exception details:")'),
    ]
    
    # Apply patches
    modified = False
    for original, replacement in original_patterns:
        if original in content:
            content = content.replace(original, replacement)
            modified = True
            logger.info(f"Applied patch to {script_path}")
    
    # Write back the modified content if changes were made
    if modified:
        backup_path = script_path + ".backup"
        # Create backup
        with open(backup_path, 'w') as f:
            f.write(open(script_path, 'r').read())
        logger.info(f"Backup created: {backup_path}")
        
        # Write patched version
        with open(script_path, 'w') as f:
            f.write(content)
        logger.info(f"Patched script: {script_path}")
        return True
    else:
        logger.info(f"No patches applied to {script_path} (pattern not found)")
        return False


def create_lora_saving_function():
    """
    Create a standalone LoRA saving function that can be manually added to scripts.
    """
    function_code = '''
def save_lora_adapters_safely(trainer, output_dir, use_lora):
    """
    Safely save LoRA adapters with proper error handling.
    
    Args:
        trainer: The trainer instance
        output_dir (str): Output directory
        use_lora (bool): Whether LoRA is enabled
    """
    if not use_lora:
        return
    
    try:
        import os
        lora_output_dir = os.path.join(output_dir, "lora_adapters")
        os.makedirs(lora_output_dir, exist_ok=True)
        
        # Use trainer.model instead of original model for LoRA saving
        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        model_to_save.save_pretrained(lora_output_dir)
        
        logger.info(f"LoRA adapters saved to {lora_output_dir}")
        
        # Verify saved files
        if os.path.exists(lora_output_dir):
            contents = os.listdir(lora_output_dir)
            logger.info(f"Contents of lora_adapters: {contents}")
            
            # Check for required files
            required_files = ["adapter_config.json"]
            has_adapter_model = any(f.startswith("adapter_model") and f.endswith((".bin", ".safetensors")) for f in contents)
            
            if "adapter_config.json" in contents and has_adapter_model:
                logger.info("LoRA adapters saved successfully with all required files")
            else:
                logger.warning("LoRA adapters directory may be missing required files")
        else:
            logger.warning(f"LoRA adapters directory was not created: {lora_output_dir}")
            
    except Exception as e:
        logger.error(f"Failed to save LoRA adapters: {e}")
        logger.exception("Exception details:")
'''
    return function_code


def main():
    parser = argparse.ArgumentParser(description="Patch Higgs Audio training scripts to fix LoRA adapter saving")
    parser.add_argument("--script", type=str, help="Path to specific trainer script to patch")
    parser.add_argument("--generate-function", action="store_true", help="Generate standalone LoRA saving function")
    parser.add_argument("--list-scripts", action="store_true", help="List available trainer scripts")
    
    # For simplicity in this standalone script, we'll handle args manually
    import argparse
    import glob
    
    # Default paths
    project_root = "/Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio"
    trainer_scripts = [
        os.path.join(project_root, "trainer", "trainer.py"),
        os.path.join(project_root, "trainer", "trainer_ddp.py"),
        os.path.join(project_root, "trainer", "train_v2_ddp.py")
    ]
    
    print("Higgs Audio LoRA Saving Patch Tool")
    print("==================================")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-function":
        print("\nStandalone LoRA saving function:")
        print(create_lora_saving_function())
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "--list-scripts":
        print("\nAvailable trainer scripts:")
        for script in trainer_scripts:
            status = "EXISTS" if os.path.exists(script) else "NOT FOUND"
            print(f"  {script} [{status}]")
        return
    
    # Default action: show info and usage
    print("\nThis tool can help fix LoRA adapter saving issues in Higgs Audio training scripts.")
    print("\nUsage:")
    print("  python lora_saving_patch.py --script PATH_TO_SCRIPT")
    print("  python lora_saving_patch.py --generate-function")
    print("  python lora_saving_patch.py --list-scripts")
    print("\nRecommended approach:")
    print("1. Use --list-scripts to see available scripts")
    print("2. Manually add the generated function to your training script")
    print("3. Replace the existing LoRA saving code with a call to save_lora_adapters_safely()")
    
    print("\nAvailable scripts:")
    for script in trainer_scripts:
        status = "EXISTS" if os.path.exists(script) else "NOT FOUND"
        print(f"  {script} [{status}]")


if __name__ == "__main__":
    main()