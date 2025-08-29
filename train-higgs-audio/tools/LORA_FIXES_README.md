# LoRA Adapter Saving Fixes

This directory contains tools to help fix LoRA adapter saving issues in Higgs Audio training scripts.

## Understanding the Issue

The LoRA adapter saving issue occurs because the training scripts sometimes try to save LoRA adapters from the original model object rather than the model managed by the trainer. The trainer's model may be wrapped (especially in DDP training) and may have the LoRA adapters properly attached, while the original model object may not.

## Tools Provided

### 1. lora_saving_example.py

A simple example demonstrating the correct approach to saving LoRA adapters:

```bash
python tools/lora_saving_example.py
```

Key points from the example:
1. Always use `trainer.model` instead of the original model
2. Handle DDP wrapped models with `hasattr(trainer.model, 'module')`
3. Include proper error handling
4. Verify that required files are saved

### 2. lora_saving_patch.py

A tool that can generate a standalone LoRA saving function or patch existing scripts:

```bash
# Generate the standalone function
python tools/lora_saving_patch.py --generate-function

# List available trainer scripts
python tools/lora_saving_patch.py --list-scripts
```

## Recommended Fix Approach

To fix LoRA adapter saving in your training scripts:

1. **Locate the LoRA saving code** in your training script (usually after `trainer.train()`)

2. **Replace the existing LoRA saving code** with the correct approach:

```python
# For DDP training (multi-GPU)
if trainer.is_world_process_zero():
    trainer.save_model()
    logger.info(f"Model saved to {args.output_dir}")
    
    # Save LoRA adapters separately with the correct approach
    if args.use_lora:
        try:
            lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
            os.makedirs(lora_output_dir, exist_ok=True)
            
            # KEY FIX: Use trainer.model instead of original model
            model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            model_to_save.save_pretrained(lora_output_dir)
            
            logger.info(f"LoRA adapters saved to {lora_output_dir}")
        except Exception as e:
            logger.error(f"Failed to save LoRA adapters: {e}")
            logger.exception("Exception details:")

# For single-GPU training
trainer.save_model()
logger.info(f"Model saved to {args.output_dir}")

# Save LoRA adapters separately with the correct approach
if args.use_lora:
    try:
        lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
        os.makedirs(lora_output_dir, exist_ok=True)
        
        # KEY FIX: Use trainer.model instead of original model
        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        model_to_save.save_pretrained(lora_output_dir)
        
        logger.info(f"LoRA adapters saved to {lora_output_dir}")
    except Exception as e:
        logger.error(f"Failed to save LoRA adapters: {e}")
        logger.exception("Exception details:")
```

## Why This Fix Works

1. **Uses the correct model object**: `trainer.model` is the model that the trainer actually trained and may have LoRA adapters attached
2. **Handles DDP wrapping**: In distributed training, the model is often wrapped in a `DistributedDataParallel` object accessible via `.module`
3. **Includes error handling**: Catches and logs any issues that occur during saving
4. **Maintains compatibility**: Follows the same patterns as the original working implementations

## Verification

After applying the fix, you can verify that LoRA adapters are saved correctly by checking that the `lora_adapters` directory contains:
- `adapter_config.json`
- `adapter_model.bin` or `adapter_model.safetensors`

These files can then be used with the `merger.py` script to merge the LoRA adapters with the base model.