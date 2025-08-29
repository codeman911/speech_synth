# LoRA Adapter Directory Fix Summary

## Problem
Users were trying to use checkpoint directories with the merger script, but LoRA adapters are saved separately in a `lora_adapters` directory. This caused errors like:
```
ValueError: Can't find weights for output_v2/checkpoint-15000 in output_v2/checkpoint-15000 or in the Hugging Face Hub.
```

## Root Cause
All training scripts (`train_v2.py`, `train_v2_ddp.py`, `trainer.py`, `trainer_ddp.py`) save LoRA adapters in a separate `lora_adapters` directory, not in checkpoint directories. This is by design for several reasons:
1. Separation of concerns - checkpoints are for resuming training, adapters are for merging
2. Storage efficiency - adapters are much smaller than full model checkpoints
3. Flexibility - same adapters can be used with different base models
4. Hugging Face PEFT convention - adapters are saved separately

## Solutions Implemented

### 1. Enhanced Error Messages in Merger Script
- Clearer error messages when users try to use checkpoint directories
- Explicit guidance on what to do instead
- Suggestions to use the `find_lora_adapters.py` helper script

### 2. Helper Script to Find LoRA Adapters
- Created `find_lora_adapters.py` to help users locate the correct directories
- Shows detailed information about found directories
- Provides usage tips when no directories are found

### 3. Improved Documentation
- Updated `README_v2.md` with clear section on LoRA adapters
- Explained directory structure and what each directory contains
- Provided clear examples of how to use the merger script

### 4. Enhanced Training Script Messaging
- Added explicit logging messages in all training scripts
- Clear instructions on how to merge LoRA adapters
- Warnings not to use checkpoint directories with merger script

## Directory Structure
```
output/
├── checkpoint-1000/          # Model checkpoint (does NOT contain LoRA adapters)
├── checkpoint-2000/          # Model checkpoint (does NOT contain LoRA adapters)
├── lora_adapters/            # LoRA adapters (this is what you need for merging)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── README.md
└── trainer_state.json
```

## Correct Usage
To merge LoRA adapters with the base model:
```bash
python trainer/merger.py \
    --base_model_path bosonai/higgs-audio-v2-generation-3B-base \
    --lora_adapter_path ./output/lora_adapters \
    --output_path ./merged_model
```

NOT this (incorrect):
```bash
python trainer/merger.py \
    --base_model_path bosonai/higgs-audio-v2-generation-3B-base \
    --lora_adapter_path ./output/checkpoint-15000 \
    --output_path ./merged_model
```

## Files Modified
1. `merger.py` - Enhanced error handling and messaging
2. `find_lora_adapters.py` - New helper script
3. `README_v2.md` - Updated documentation
4. `train_v2.py` - Enhanced logging messages
5. `train_v2_ddp.py` - Enhanced logging messages
6. `trainer.py` - Enhanced logging messages
7. `trainer_ddp.py` - Enhanced logging messages

## Enhanced Debugging for LoRA Adapter Saving

To help debug issues with LoRA adapter saving, enhanced logging has been added to both training scripts:

1. **Detailed logging during LoRA setup**: After applying LoRA configuration, the scripts now log detailed information about the model structure
2. **Detailed logging during LoRA saving**: Before and during the saving process, the scripts log information about the model being saved and any errors that occur

If you're experiencing issues with LoRA adapters not being saved, check the logs for:
- "LoRA flag is set, attempting to save LoRA adapters..."
- "Model to save type: ..."
- Any error messages that might indicate why saving is failing

These enhanced logs will help identify whether the issue is with:
- The LoRA configuration not being applied correctly
- The model not being properly converted to a LoRA model
- Issues during the saving process
