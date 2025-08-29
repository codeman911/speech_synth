#!/usr/bin/env python3
"""
Verification script to confirm that train_v2_ddp.py already uses the correct LoRA saving approach.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_train_v2_ddp():
    """
    Analyze train_v2_ddp.py to confirm it uses the correct LoRA saving approach.
    """
    project_root = "/Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio"
    script_path = os.path.join(project_root, "trainer", "train_v2_ddp.py")
    
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
    
    print("Analyzing train_v2_ddp.py LoRA Saving Implementation")
    print("====================================================")
    
    # Read the script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for the correct LoRA saving approach
    correct_approach = "model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model"
    
    if correct_approach in content:
        print("✓ train_v2_ddp.py ALREADY uses the correct approach:")
        print("  model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model")
        print("  model_to_save.save_pretrained(lora_output_dir)")
        return True
    else:
        print("✗ train_v2_ddp.py does not contain the expected pattern")
        return False


def compare_with_other_scripts():
    """
    Compare the LoRA saving approach across different training scripts.
    """
    project_root = "/Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio"
    
    scripts = {
        "train_v2_ddp.py": os.path.join(project_root, "trainer", "train_v2_ddp.py"),
        "trainer_ddp.py": os.path.join(project_root, "trainer", "trainer_ddp.py"),
        "trainer.py": os.path.join(project_root, "trainer", "trainer.py")
    }
    
    print("\n\nComparison of LoRA Saving Approaches")
    print("====================================")
    
    # Check each script
    for script_name, script_path in scripts.items():
        if not os.path.exists(script_path):
            print(f"\n{script_name}: NOT FOUND")
            continue
            
        with open(script_path, 'r') as f:
            content = f.read()
        
        print(f"\n{script_name}:")
        
        # Check for train_v2_ddp.py approach
        if "model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model" in content:
            print("  ✓ Uses trainer.model approach (CORRECT)")
        
        # Check for trainer.py approach
        elif "model.model.text_model.save_pretrained" in content:
            print("  ✗ Uses model.model.text_model approach (POTENTIALLY PROBLEMATIC)")
            
        # Check for simple trainer.model approach
        elif "trainer.model.save_pretrained" in content and "hasattr(trainer.model, 'module')" not in content:
            print("  ⚠ Uses simple trainer.model approach (MAY MISS DDP WRAPPING)")
            
        else:
            print("  ? Uses unknown approach")


def main():
    print("Higgs Audio LoRA Saving Verification")
    print("===================================")
    
    # Analyze train_v2_ddp.py specifically
    is_correct = analyze_train_v2_ddp()
    
    # Compare with other scripts
    compare_with_other_scripts()
    
    print("\n\nConclusion:")
    print("===========")
    if is_correct:
        print("✓ train_v2_ddp.py already uses the correct LoRA saving approach")
        print("  It follows the same pattern as trainer_ddp.py which is known to work")
        print("  No changes needed for this script")
    else:
        print("✗ Unexpected: train_v2_ddp.py does not use the expected approach")
        print("  This may indicate a different version or modification")
    
    print("\nThe correct approach is:")
    print("  model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model")
    print("  model_to_save.save_pretrained(lora_output_dir)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())