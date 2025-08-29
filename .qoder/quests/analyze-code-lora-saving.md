# LoRA Adapter Saving Issue in train_v2_ddp.py

## Problem

The `train_v2_ddp.py` script has code to save LoRA adapters, but they are not being saved correctly.

## Root Cause

The script attempts to call `save_pretrained` directly on the model without checking the model structure:

```python
model_to_save.save_pretrained(lora_output_dir)
```

However, the working implementations in `train_v2.py` and `trainer_ddp.py` properly check the model structure before calling `save_pretrained`:

```python
if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
    model.model.text_model.save_pretrained(lora_output_dir)
elif hasattr(model, 'model'):
    model.model.save_pretrained(lora_output_dir)
else:
    model.save_pretrained(lora_output_dir)
```

This is necessary because the HiggsAudioModelWrapper class wraps the actual model, and the PEFT LoRA model is nested inside it.

## Solution

Update the LoRA saving logic in `train_v2_ddp.py` to properly check the model structure before calling `save_pretrained`, matching the pattern used in the working implementations.

## Implementation Plan

1. Modify the LoRA saving section in `train_v2_ddp.py` to properly check the model structure
2. Add try/except error handling around the saving operation
3. Add informative logging to help with debugging
4. Add verification messages to confirm successful saving

## Code Changes Required

The specific changes needed in `train_v2_ddp.py` are:

1. Replace the simple `model_to_save.save_pretrained(lora_output_dir)` call with the proper model structure checking logic
2. Add error handling and informative logging

## Verification

After implementing these changes, the LoRA adapters should be properly saved to the `lora_adapters` subdirectory, and appropriate logging should confirm this.