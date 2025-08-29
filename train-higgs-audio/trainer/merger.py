#!/usr/bin/env python3
"""
LoRA Merger for Higgs Audio v2
Merges LoRA adapters back into the base model
"""

import os
import torch
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig
import json
import datetime
import shutil 

try:
    from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
    HIGGS_AVAILABLE = True
    logging.info("Successfully imported Higgs Audio specific modules.")
except ImportError:
    HIGGS_AVAILABLE = False
    from transformers import AutoModel # Fallback, though likely to fail for 'higgs_audio'
    logging.warning(
        "Could not import Higgs Audio modules from 'boson_multimodal'. "
        "Falling back to `AutoModel`. This will likely fail if the model "
        "type 'higgs_audio' is not registered with Transformers."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HiggsAudioLoRaMerger:
    """Merger for LoRA adapters with Higgs Audio v2 models"""
    
    def __init__(self, base_model_path: str, lora_adapter_path: str):
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        
    def load_base_model(self):
        """Load the base Higgs Audio model"""
        logger.info(f"Loading base model from {self.base_model_path}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, 
            trust_remote_code=True
        )
        
        # 确保 tokenizer 有一个 padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer's pad_token was None, set to eos_token: {self.tokenizer.pad_token}")
        
        # 修改加载模型的方式，不再使用通用的 AutoModel，
        # 而是使用从 boson_multimodal 导入的特定 HiggsAudioModel 类。
        if HIGGS_AVAILABLE:
            logger.info("Using specific HiggsAudioModel and HiggsAudioConfig to load the base model.")
            config = HiggsAudioConfig.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            self.base_model = HiggsAudioModel.from_pretrained(
                self.base_model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            logger.error("The `boson_multimodal` library is required to load the 'higgs_audio' model type but it was not found.")
            raise ImportError("Please install the required 'boson_multimodal' library to proceed.")
        
        logger.info("Base model loaded successfully")
        return self.base_model
        
    def load_lora_model(self):
        """Load model with LoRA adapters"""
        logger.info(f"Loading LoRA adapters from {self.lora_adapter_path}")
        
        # Normalize the path to avoid issues with trailing slashes
        lora_adapter_path = os.path.normpath(self.lora_adapter_path)
        
        # Check if this is a checkpoint directory and suggest the correct lora_adapters path
        if "checkpoint-" in lora_adapter_path:
            # Look for lora_adapters directory in the parent directory
            parent_dir = os.path.dirname(lora_adapter_path)
            lora_adapters_dir = os.path.join(parent_dir, "lora_adapters")
            if os.path.exists(lora_adapters_dir):
                logger.error("CRITICAL ERROR: You are trying to use a CHECKPOINT directory!")
                logger.error("LoRA adapters are SAVED SEPARATELY in a 'lora_adapters' directory.")
                logger.error(f"Use this path instead: {lora_adapters_dir}")
                logger.error("DO NOT use checkpoint directories for merging LoRA adapters!")
                raise ValueError(f"Invalid path: {lora_adapter_path} is a checkpoint directory, not a LoRA adapters directory. "
                               f"Use {lora_adapters_dir} instead.")
            else:
                # Check if lora_adapters exists in the output directory
                output_dir = parent_dir
                while output_dir and "checkpoint-" in os.path.basename(output_dir):
                    output_dir = os.path.dirname(output_dir)
                if output_dir and os.path.exists(os.path.join(output_dir, "lora_adapters")):
                    suggested_path = os.path.join(output_dir, "lora_adapters")
                    logger.error("CRITICAL ERROR: You are trying to use a CHECKPOINT directory!")
                    logger.error("LoRA adapters are SAVED SEPARATELY in a 'lora_adapters' directory.")
                    logger.error(f"Use this path instead: {suggested_path}")
                    logger.error("DO NOT use checkpoint directories for merging LoRA adapters!")
                    raise ValueError(f"Invalid path: {lora_adapter_path} is a checkpoint directory, not a LoRA adapters directory. "
                                   f"Use {suggested_path} instead.")
                else:
                    logger.error("CRITICAL ERROR: You are trying to use a CHECKPOINT directory!")
                    logger.error("LoRA adapters are SAVED SEPARATELY in a 'lora_adapters' directory.")
                    logger.error("Please check your training output directory for a 'lora_adapters' folder.")
                    logger.error("DO NOT use checkpoint directories for merging LoRA adapters!")
                    # Show how to find lora_adapters directories
                    logger.error("To find the correct lora_adapters directory, run:")
                    logger.error("  python find_lora_adapters.py --path /path/to/your/training/output")
                    raise ValueError(f"Invalid path: {lora_adapter_path} is a checkpoint directory, not a LoRA adapters directory.")
        
        # Check if adapter_config.json exists
        adapter_config_path = Path(lora_adapter_path) / "adapter_config.json"
        if not adapter_config_path.exists():
            logger.error(f"adapter_config.json not found in {lora_adapter_path}.")
            logger.error("This directory does not appear to contain LoRA adapters.")
            logger.error("LoRA adapters are saved in a separate 'lora_adapters' directory, not in checkpoint directories.")
            # Show how to find lora_adapters directories
            logger.error("To find the correct lora_adapters directory, run:")
            logger.error("  python find_lora_adapters.py --path /path/to/your/training/output")
            raise ValueError(f"adapter_config.json not found in {lora_adapter_path}. "
                           f"This directory does not appear to contain LoRA adapters. "
                           f"LoRA adapters are saved in a separate 'lora_adapters' directory, "
                           f"not in checkpoint directories.")
        
        # Check if adapter model files exist
        adapter_model_files = [
            os.path.join(lora_adapter_path, "adapter_model.bin"),
            os.path.join(lora_adapter_path, "adapter_model.safetensors")
        ]
        adapter_model_exists = any(os.path.exists(f) for f in adapter_model_files)
        
        if not adapter_model_exists:
            logger.error(f"Neither adapter_model.bin nor adapter_model.safetensors found in {lora_adapter_path}.")
            logger.error("This directory does not contain valid LoRA adapter files.")
            # Show how to find lora_adapters directories
            logger.error("To find the correct lora_adapters directory, run:")
            logger.error("  python find_lora_adapters.py --path /path/to/your/training/output")
            raise ValueError(f"Neither adapter_model.bin nor adapter_model.safetensors found in {lora_adapter_path}. "
                           f"This directory does not contain valid LoRA adapter files.")
        
        # Load LoRA config with local_files_only=True to avoid remote repo issues
        try:
            peft_config = PeftConfig.from_pretrained(lora_adapter_path, local_files_only=True)
        except Exception as e:
            logger.warning(f"Failed to load config with local_files_only=True: {e}. Trying without it.")
            peft_config = PeftConfig.from_pretrained(lora_adapter_path)
        logger.info(f"LoRA config: {peft_config}")
        
        # Load model with LoRA adapters
        try:
            self.lora_model = PeftModel.from_pretrained(
                self.base_model,
                lora_adapter_path,
                config=peft_config,
                is_trainable=False,  # Make sure we're loading for inference
                local_files_only=True  # Avoid remote repo issues
            )
        except Exception as e:
            logger.warning(f"Failed to load model with local_files_only=True: {e}. Trying without it.")
            self.lora_model = PeftModel.from_pretrained(
                self.base_model,
                lora_adapter_path,
                config=peft_config,
                is_trainable=False  # Make sure we're loading for inference
            )
        
        logger.info("LoRA model loaded successfully")
        return self.lora_model
        
    def merge_and_save(self, output_path: str, save_tokenizer: bool = True):
        """Merge LoRA weights and save the merged model"""
        logger.info("Merging LoRA weights into base model...")
        
        # Merge LoRA weights
        merged_model = self.lora_model.merge_and_unload()
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save merged model (this should typically handle generation_config if it's a PreTrainedModel)
        logger.info(f"Saving merged model to {output_path}")
        merged_model.save_pretrained(
            output_path,
            # safe_serialization=True # 某些自定义模型可能不支持，如果保存失败可以注释掉
        )
        
        # 显式保存 generation_config.json
        if hasattr(merged_model, 'generation_config') and merged_model.generation_config is not None:
            logger.info(f"Saving generation_config.json from merged model to {output_path}")
            # GenerationConfig 对象有自己的 save_pretrained 方法
            merged_model.generation_config.save_pretrained(output_path)
        else:
            logger.warning("No generation_config found on the merged model. Attempting to copy from base model path.")
            # 如果模型上没有 generation_config 属性，尝试从原始模型路径复制
            base_gen_config_path = Path(self.base_model_path) / "generation_config.json"
            if base_gen_config_path.exists():
                shutil.copy(base_gen_config_path, output_dir / "generation_config.json")
                logger.info(f"Copied generation_config.json from base model path: {base_gen_config_path}")
            else:
                logger.warning(f"No generation_config.json found in base model path: {base_gen_config_path}. Generation config will not be saved.")

        # 保存 tokenizer 和相关文件
        if save_tokenizer:
            logger.info(f"Saving tokenizer and its configuration files to {output_path}")
            self.tokenizer.save_pretrained(output_path)
            
            # 只有当目标路径不存在该文件时才复制，避免覆盖。
            tokenizer_files_to_copy = [
                "special_tokens_map.json",
                "tokenizer_config.json",
                "tokenizer.json",  # 对于 fast tokenizers
                "vocab.json",      # 对于 slow tokenizers (例如 BERT/GPT-2)
                "added_tokens.json", # 如果有额外添加的 token
            ]
            
            for file_name in tokenizer_files_to_copy:
                src_path = Path(self.base_model_path) / file_name
                dest_path = output_dir / file_name
                
                if src_path.exists() and not dest_path.exists():
                    try:
                        shutil.copy(src_path, dest_path)
                        logger.info(f"Copied missing tokenizer file '{file_name}' from base model path to output directory.")
                    except Exception as e:
                        logger.warning(f"Failed to copy '{file_name}' from '{src_path}' to '{dest_path}': {e}")
                elif not src_path.exists():
                    logger.debug(f"Source tokenizer file '{file_name}' not found at '{src_path}'. Skipping copy.")
                else:
                    logger.debug(f"Tokenizer file '{file_name}' already exists in output directory or not needed for copy.")
            
        # Save model info
        model_info = {
            "base_model": self.base_model_path,
            "lora_adapter": self.lora_adapter_path,
            "merged_at": datetime.datetime.utcnow().isoformat(),
            "model_type": "higgs-audio-v2-lora-merged"
        }
        
        with open(output_dir / "merge_info.json", 'w') as f:
            json.dump(model_info, f, indent=4)
            
        logger.info(f"Model merged and saved successfully to {output_path}")
        
    def compare_models(self, test_input: str = None):
        """Compare outputs between base model and LoRA model"""
        if test_input is None:
            test_input = "The sun rises in the east and sets in the west."
            
        logger.info(f"Comparing base model and LoRA model outputs using input: '{test_input}'")
        
        # Tokenize input
        inputs = self.tokenizer(
            test_input,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.base_model.device) # 确保输入在模型所在的设备上
        
        # Get base model output
        with torch.no_grad():
            base_outputs = self.base_model(**inputs)
            
        # Get LoRA model output  
        with torch.no_grad():
            lora_outputs = self.lora_model(**inputs)
            
        # Compare outputs
        diff = None
        if hasattr(base_outputs, 'last_hidden_state') and hasattr(lora_outputs, 'last_hidden_state'):
            base_tensor = base_outputs.last_hidden_state
            lora_tensor = lora_outputs.last_hidden_state
            tensor_name = "last_hidden_state"
        elif hasattr(base_outputs, 'logits') and hasattr(lora_outputs, 'logits'):
            base_tensor = base_outputs.logits
            lora_tensor = lora_outputs.logits
            tensor_name = "logits"
        else:
            logger.warning("Could not find 'last_hidden_state' or 'logits' in model outputs to compare.")
            return {}

        diff = torch.abs(base_tensor - lora_tensor).mean().item()
        logger.info(f"Average absolute difference in '{tensor_name}': {diff:.6f}")
        
        base_mean = torch.abs(base_tensor).mean().item()
        if base_mean > 1e-9:
            rel_change = (diff / base_mean) * 100
            logger.info(f"Relative change: {rel_change:.2f}%")
        else:
            logger.info("Base tensor mean is close to zero, relative change is not meaningful.")

        return {
            "base_output": base_outputs,
            "lora_output": lora_outputs,
            "difference": diff
        }

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with Higgs Audio v2 base model")
    
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="Path to the local base Higgs Audio model directory"
    )
    parser.add_argument(
        "--lora_adapter_path", 
        type=str, 
        required=True,
        help="Path to LoRA adapter directory (NOT a checkpoint directory - use the separate 'lora_adapters' directory)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--compare_models", 
        action="store_true",
        help="Compare base and LoRA models before merging"
    )
    parser.add_argument(
        "--test_input", 
        type=str,
        default="The sun rises in the east and sets in the west.",
        help="Test input for model comparison"
    )
    parser.add_argument(
        '--no_save_tokenizer', 
        action='store_false', 
        dest='save_tokenizer',
        help="Do not save the tokenizer with the merged model"
    )
    parser.set_defaults(save_tokenizer=True)
    
    args = parser.parse_args()
    
    # Normalize paths
    args.lora_adapter_path = os.path.normpath(args.lora_adapter_path)
    args.base_model_path = os.path.normpath(args.base_model_path)
    args.output_path = os.path.normpath(args.output_path)
    
    # Validate paths
    if not os.path.exists(args.lora_adapter_path):
        logger.error(f"LoRA adapter path does not exist: {args.lora_adapter_path}")
        logger.error("To find the correct lora_adapters directory, run:")
        logger.error("  python find_lora_adapters.py --path /path/to/your/training/output")
        raise ValueError(f"LoRA adapter path does not exist: {args.lora_adapter_path}")
    
    # Additional validation to ensure we have the required files
    adapter_config_path = Path(args.lora_adapter_path) / "adapter_config.json"
    adapter_model_files = [
        os.path.join(args.lora_adapter_path, "adapter_model.bin"),
        os.path.join(args.lora_adapter_path, "adapter_model.safetensors")
    ]
    adapter_model_exists = any(os.path.exists(f) for f in adapter_model_files)
    
    if not adapter_config_path.exists():
        logger.error(f"adapter_config.json not found in {args.lora_adapter_path}.")
        logger.error("This directory does not appear to contain LoRA adapters.")
        logger.error("LoRA adapters are saved in a separate 'lora_adapters' directory, not in checkpoint directories.")
        logger.error("To find the correct lora_adapters directory, run:")
        logger.error("  python find_lora_adapters.py --path /path/to/your/training/output")
        raise ValueError(f"adapter_config.json not found in {args.lora_adapter_path}. "
                       f"This directory does not appear to contain LoRA adapters. "
                       f"LoRA adapters are saved in a separate 'lora_adapters' directory, "
                       f"not in checkpoint directories.")
    
    if not adapter_model_exists:
        logger.error(f"Neither adapter_model.bin nor adapter_model.safetensors found in {args.lora_adapter_path}.")
        logger.error("This directory does not contain valid LoRA adapter files.")
        logger.error("To find the correct lora_adapters directory, run:")
        logger.error("  python find_lora_adapters.py --path /path/to/your/training/output")
        raise ValueError(f"Neither adapter_model.bin nor adapter_model.safetensors found in {args.lora_adapter_path}. "
                       f"This directory does not contain valid LoRA adapter files.")
        
    # Initialize merger
    merger = HiggsAudioLoRaMerger(args.base_model_path, args.lora_adapter_path)
    
    # Load models
    merger.load_base_model()
    merger.load_lora_model()
    
    # Compare models if requested
    if args.compare_models:
        merger.compare_models(args.test_input)
        logger.info("Model comparison completed")
        
    # Merge and save
    merger.merge_and_save(args.output_path, args.save_tokenizer)
    
    logger.info("LoRA merge process completed successfully!")

if __name__ == "__main__":
    main()