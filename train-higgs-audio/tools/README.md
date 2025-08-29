# Higgs Audio Training Tools

This directory contains utility scripts for working with Higgs Audio training.

## LoRA Adapter Saving Fixes

### Simple Solution (Recommended)
- [LORA_FIX_SUMMARY.md](LORA_FIX_SUMMARY.md) - Simple fix using the proven approach from trainer_ddp.py
- [fix_lora_saving_simple.py](fix_lora_saving_simple.py) - Shows the exact code to apply

### train_v2_ddp.py Alignment
- [TRAIN_V2_DDP_ALIGNMENT.md](TRAIN_V2_DDP_ALIGNMENT.md) - Confirmation that train_v2_ddp.py already uses the correct approach
- [train_v2_ddp_lora_fix.py](train_v2_ddp_lora_fix.py) - Analysis of train_v2_ddp.py implementation

### Comprehensive Solutions
- [LORA_FIXES_README.md](LORA_FIXES_README.md) - Documentation for fixing LoRA adapter saving issues
- [lora_saving_example.py](lora_saving_example.py) - Example demonstrating correct LoRA saving approach
- [lora_saving_patch.py](lora_saving_patch.py) - Tool to generate or patch LoRA saving code
- [verify_lora_adapters.py](verify_lora_adapters.py) - Utility to verify LoRA adapters were saved correctly
- [fix_lora_saving.py](fix_lora_saving.py) - Standalone utility to extract LoRA adapters from checkpoints

### Analysis Tools
- [analyze_lora_approaches.py](analyze_lora_approaches.py) - Compares different LoRA saving approaches
- [verify_train_v2_ddp_lora.py](verify_train_v2_ddp_lora.py) - Verifies train_v2_ddp.py uses correct approach
- [LORA_SAVING_FIX_SUMMARY.md](LORA_SAVING_FIX_SUMMARY.md) - Detailed analysis of the fix
