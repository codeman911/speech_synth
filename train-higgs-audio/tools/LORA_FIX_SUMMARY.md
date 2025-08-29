# LoRA Adapter Saving Fix - Simple Solution

## Problem
The LoRA adapter saving issue in Higgs Audio training scripts occurs because the scripts try to save LoRA adapters from the original model object rather than the model managed by the trainer.

## Root Cause
- **trainer.py** uses `model.save_pretrained()` - the original model object
- **trainer_ddp.py** uses `trainer.model.save_pretrained()` - the trainer's model object
- The trainer's model is the one that actually has the LoRA adapters attached

## Solution
Use the simple, proven approach from [trainer_ddp.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/trainer/trainer_ddp.py):

```python
# Replace the existing LoRA saving code with this:
if args.use_lora:
    lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
    # Use trainer.model instead of original model for LoRA saving
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    model_to_save.save_pretrained(lora_output_dir)
    logger.info(f"LoRA adapters saved to {lora_output_dir}")
```

## For DDP Training Scripts
Wrap in the world process check:
```python
if trainer.is_world_process_zero():
    trainer.save_model()
    logger.info(f"Model saved to {args.output_dir}")
    
    # Save LoRA adapters separately
    if args.use_lora:
        lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        model_to_save.save_pretrained(lora_output_dir)
        logger.info(f"LoRA adapters saved to {lora_output_dir}")
```

## For Single-GPU Training Scripts
```python
trainer.save_model()
logger.info(f"Model saved to {args.output_dir}")

# Save LoRA adapters separately
if args.use_lora:
    lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    model_to_save.save_pretrained(lora_output_dir)
    logger.info(f"LoRA adapters saved to {lora_output_dir}")
```

## Why This Works
1. **Uses the correct model**: `trainer.model` is the model that was actually trained
2. **Handles DDP wrapping**: The `hasattr` check properly accesses wrapped models
3. **Proven approach**: This is the exact pattern that works in [trainer_ddp.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/trainer/trainer_ddp.py)
4. **Simple and reliable**: No complex model structure assumptions

## Verification
After applying the fix, check that your output contains:
```
lora_adapters/
├── adapter_config.json
└── adapter_model.bin (or adapter_model.safetensors)
```

Both files should be non-zero in size, indicating the LoRA weights were saved correctly.

## Tools Provided
- [fix_lora_saving_simple.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/tools/fix_lora_saving_simple.py) - Shows the exact code to use
- [analyze_lora_approaches.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/tools/analyze_lora_approaches.py) - Compares the different approaches