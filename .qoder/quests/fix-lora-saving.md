# Fix LoRA Adapter Saving Issue in Higgs Audio Training Scripts

## Overview

The LoRA adapter saving functionality in the Higgs Audio training scripts is not working correctly. When training with LoRA enabled, the adapters are not being saved properly to the `lora_adapters` directory during checkpoint saving. This issue affects both the single-GPU (`trainer.py`) and multi-GPU DDP (`trainer_ddp.py`, `train_v2_ddp.py`) training scripts.

## Problem Analysis

### Current Implementation Issues

1. **Inadequate LoRA Saving Logic**: The current implementation in `train_v2_ddp.py` has extensive logging but still fails to save LoRA adapters correctly.

2. **Incorrect Model Reference**: The scripts are not correctly identifying the PEFT model component that needs to be saved.

3. **Checkpoint vs. Adapter Separation**: There's confusion between saving full model checkpoints and LoRA adapters separately.

### Root Causes

1. The model structure after applying LoRA is complex, with potential nested modules
2. The saving logic doesn't correctly identify which part of the model contains the LoRA adapters
3. The `save_pretrained` method is not being called on the correct model component

## Solution Design

### Key Principles

Following the LoRA Adapter Saving Implementation Guidelines:
1. Always call `trainer.save_model()` first to ensure proper checkpoint saving
2. Then separately save LoRA adapters when using LoRA
3. Adapters should be saved separately in a 'lora_adapters' subdirectory
4. Include comprehensive logging to verify the saving process

### Implementation Approach

1. **Enhanced Model Detection**: Implement a robust method to identify the PEFT model component
2. **Corrected Saving Logic**: Ensure `save_pretrained` is called on the right model component
3. **Improved Error Handling**: Add comprehensive error handling and logging

## Detailed Implementation

### 1. Enhanced Model Detection Function

```python
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
```

### 2. Improved LoRA Saving Function

```python
def save_lora_adapters(trainer, output_dir, use_lora):
    """
    Save LoRA adapters separately from model checkpoints
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
    
    try:
        # Save the LoRA adapters
        lora_model.save_pretrained(lora_output_dir)
        logger.info(f"LoRA adapters successfully saved to {lora_output_dir}")
    except Exception as e:
        logger.error(f"Failed to save LoRA adapters: {e}")
        logger.exception("Exception details:")
```

### 3. Integration with Training Scripts

For all training scripts (`trainer.py`, `trainer_ddp.py`, `train_v2_ddp.py`):

```python
# In the main training function, after trainer.train()
if trainer.is_world_process_zero():
    # Save the main model checkpoint
    trainer.save_model()
    logger.info(f"Model checkpoint saved to {args.output_dir}")
    
    # Save LoRA adapters separately
    save_lora_adapters(trainer, args.output_dir, args.use_lora)
```

## Architecture Changes

### Model Structure Understanding

```mermaid
graph TD
    A[HiggsAudioModelWrapper] --> B[DDP Wrapper (optional)]
    B --> C[Actual Model]
    C --> D[HiggsAudioModel]
    D --> E[text_model with LoRA]
    
    style E fill:#cde4ff,stroke:#6495ED,stroke-width:2px
```

The LoRA adapters are applied to the `text_model` component, which needs to be correctly identified for saving.

## Implementation Steps

1. **Add the helper functions** to identify and save LoRA models
2. **Modify the saving logic** in all training scripts to use the new approach
3. **Add comprehensive logging** to verify each step of the process
4. **Test with both single-GPU and DDP training** to ensure compatibility

## Testing Plan

1. **Single-GPU Training Test**:
   - Run training with LoRA enabled using `trainer.py`
   - Verify that `lora_adapters` directory is created with proper files

2. **DDP Training Test**:
   - Run training with LoRA enabled using `train_v2_ddp.py`
   - Verify that `lora_adapters` directory is created with proper files

3. **File Verification**:
   - Check for presence of `adapter_config.json`
   - Check for presence of `adapter_model.bin` or `adapter_model.safetensors`

## Expected Outcomes

1. LoRA adapters will be correctly saved to the `lora_adapters` subdirectory
2. The saved adapters can be used for model merging with the `merger.py` script
3. Training checkpoints and LoRA adapters are properly separated
4. Comprehensive logging will help diagnose any future issues

## Rollback Plan

If issues occur:
1. Revert to the previous saving implementation
2. Use the working pattern from `trainer_ddp.py` as a reference
3. Ensure training is not disrupted while fixing the saving issue