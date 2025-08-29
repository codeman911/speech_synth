# Alignment with train_v2_ddp.py

## Current Status

**✓ train_v2_ddp.py already uses the correct LoRA saving approach**

The [train_v2_ddp.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/trainer/train_v2_ddp.py) script already implements the recommended LoRA saving approach:

```python
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
model_to_save.save_pretrained(lora_output_dir)
```

This is exactly the same approach used in [trainer_ddp.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/trainer/trainer_ddp.py) which is known to work correctly.

## Verification

1. **Correct Model Reference**: Uses `trainer.model` instead of the original model object
2. **DDP Handling**: Properly handles DDP wrapping with `hasattr(trainer.model, 'module')`
3. **Error Handling**: Includes try/except blocks for robust error handling
4. **Logging**: Has extensive logging to diagnose issues
5. **File Verification**: Checks directory creation and file contents

## Comparison with Other Scripts

| Script | LoRA Saving Approach | Status |
|--------|---------------------|--------|
| train_v2_ddp.py | `trainer.model.module if hasattr(trainer.model, 'module') else trainer.model` | ✅ **CORRECT** |
| trainer_ddp.py | `trainer.model.module if hasattr(trainer.model, 'module') else trainer.model` | ✅ **CORRECT** |
| trainer.py | `model.model.text_model` fallback chain | ⚠️ **POTENTIALLY PROBLEMATIC** |

## If You're Experiencing Issues

Since [train_v2_ddp.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/trainer/train_v2_ddp.py) already uses the correct approach, any issues are likely due to:

1. **Environment Issues**: Missing dependencies or incorrect versions
2. **Model Loading Problems**: Issues with loading the base model
3. **File Permissions**: Insufficient permissions to write to output directory
4. **Disk Space**: Insufficient disk space for saving adapters
5. **PyTorch/PEFT Issues**: Version incompatibilities

## Recommended Actions

1. **Check the logs** from your train_v2_ddp.py run to see specific error messages
2. **Verify dependencies** are correctly installed:
   ```bash
   pip list | grep -E "(torch|peft|transformers)"
   ```
3. **Check file permissions** on your output directory
4. **Ensure sufficient disk space** is available

## Minor Improvement (Optional)

If you want to slightly improve the robustness, you can simplify the logging and add better file verification:

```python
if args.use_lora:
    try:
        lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
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
            logger.error(f"LoRA adapters directory was not created: {lora_output_dir}")
            
    except Exception as e:
        logger.error(f"Failed to save LoRA adapters: {e}")
        logger.exception("Exception details:")
```

## Conclusion

**No changes are needed to [train_v2_ddp.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/trainer/train_v2_ddp.py) for LoRA saving** - it already uses the correct approach that aligns with [trainer_ddp.py](file:///Users/vikram.solanki/Projects/exp/level1/speech_synth/train-higgs-audio/trainer/trainer_ddp.py). Any issues you're experiencing are likely due to environmental factors rather than the LoRA saving implementation.