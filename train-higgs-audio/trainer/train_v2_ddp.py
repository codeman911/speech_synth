#!/usr/bin/env python3
"""
Higgs Audio v2 Training Script for Zero-Shot Voice Cloning with ChatML Dataset (DDP Version)
Based on the Higgs Audio v2 architecture from Boson AI

This script implements a training pipeline specifically compatible with zero-shot voice cloning 
datasets in ChatML format for multi-GPU training with torchrun. The implementation follows the 
same data processing pipeline as the inference script to ensure consistency between training and inference.
"""

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import librosa
import re
from dataclasses import asdict

# Try to import Higgs Audio related modules
try:
    from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator, HiggsAudioBatchInput
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
    from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
    from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
    HIGGS_AVAILABLE = True
except ImportError:
    HIGGS_AVAILABLE = False
    logging.warning("Higgs Audio modules not available. Using fallback implementation.")

    # A simple fallback class is sufficient. It does NOT need a .to() method.
    class ChatMLDatasetSample:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add constants
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

ZERO_SHOT_DEFAULT_SYSTEM_MESSAGE = """You are a helpful assistant capable of generating speech in the voice of the provided reference audio."""


class ExtendedHiggsAudioBatchInput:
    """
    Extended HiggsAudioBatchInput with __len__ method for Trainer compatibility
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        """Return the batch size based on input_ids"""
        if hasattr(self, 'input_ids') and self.input_ids is not None:
            return self.input_ids.shape[0]
        else:
            return 0

    def __getitem__(self, key):
        """Allow dictionary-style access for compatibility"""
        return getattr(self, key)

    def __contains__(self, key):
        """Check if attribute exists"""
        return hasattr(self, key)

    def keys(self):
        """Return all attribute names for compatibility"""
        return [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]


class ExtendedHiggsAudioSampleCollator:
    """
    Extended collator that returns our custom batch input class
    Configured to match the inference script exactly
    """
    def __init__(self, **kwargs):
        if HIGGS_AVAILABLE:
            # Assuming HiggsAudioSampleCollator can be initialized with these kwargs
            self.base_collator = HiggsAudioSampleCollator(**kwargs)
        else:
            # Fallback collator
            self.pad_token_id = kwargs.get('pad_token_id', 0)

    def __call__(self, batch: List[ChatMLDatasetSample]):
        if HIGGS_AVAILABLE and hasattr(self, 'base_collator'):
            batch_input = self.base_collator(batch)
            label_audio_ids = batch_input.audio_out_ids
            extended_batch = ExtendedHiggsAudioBatchInput(
                input_ids=batch_input.input_ids,
                attention_mask=batch_input.attention_mask,
                audio_features=batch_input.audio_features,
                audio_feature_attention_mask=batch_input.audio_feature_attention_mask,
                audio_out_ids=batch_input.audio_out_ids,
                audio_out_ids_start=batch_input.audio_out_ids_start,
                audio_out_ids_start_group_loc=batch_input.audio_out_ids_start_group_loc,
                audio_in_ids=batch_input.audio_in_ids,
                audio_in_ids_start=batch_input.audio_in_ids_start,
                label_ids=batch_input.label_ids,
                label_audio_ids=label_audio_ids,
                reward=batch_input.reward,
            )
            return extended_batch
        else:
            # Fallback implementation
            input_ids_list, attention_mask_list, label_ids_list = [], [], []
            for sample in batch:
                input_ids_list.append(sample.input_ids)
                attention_mask_list.append(torch.ones_like(sample.input_ids))
                label_ids_list.append(getattr(sample, 'label_ids', sample.input_ids))

            max_len = max(len(ids) for ids in input_ids_list)
            padded_input_ids, padded_attention_mask, padded_label_ids = [], [], []

            for i in range(len(input_ids_list)):
                pad_len = max_len - len(input_ids_list[i])
                padded_input_ids.append(torch.cat([input_ids_list[i], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
                padded_attention_mask.append(torch.cat([attention_mask_list[i], torch.zeros(pad_len, dtype=torch.long)]))
                padded_label_ids.append(torch.cat([label_ids_list[i], torch.full((pad_len,), -100, dtype=torch.long)]))

            return ExtendedHiggsAudioBatchInput(
                input_ids=torch.stack(padded_input_ids),
                attention_mask=torch.stack(padded_attention_mask),
                label_ids=torch.stack(padded_label_ids),
                audio_features=None, audio_feature_attention_mask=None,
                audio_out_ids=None, audio_out_ids_start=None, audio_out_ids_start_group_loc=None,
                audio_in_ids=None, audio_in_ids_start=None, label_audio_ids=None, reward=None,
            )

def normalize_chinese_punctuation(text):
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!",
        "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
        """: '"', """: '"', "'": "'", "'": "'", "、": ",", "--": "-",
        "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text

def _build_system_message_with_audio_prompt(system_message):
    contents = []
    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN):]
    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    return Message(role="system", content=contents)


class ZeroShotVoiceCloningDataset(Dataset):
    """
    Dataset class for zero-shot voice cloning training with ChatML format data.
    This implementation follows the exact same pattern as the inference script.
    """
    
    def __init__(
        self, 
        data_file: str,
        tokenizer: AutoTokenizer,
        audio_tokenizer,
        sample_rate: int = 24000,
    ):
        """
        Initialize the zero-shot voice cloning dataset.
        
        Args:
            data_file: Path to ChatML JSON file
            tokenizer: Text tokenizer
            audio_tokenizer: Audio tokenizer
            sample_rate: Target sample rate for audio processing
        """
        self.data_file = Path(data_file)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.sample_rate = sample_rate
        
        # Load samples from ChatML file
        self.samples = self._load_samples_from_chatml()
            
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {data_file}")
            
        logger.info(f"Loaded {len(self.samples)} samples from {data_file}")

    def _load_samples_from_chatml(self) -> List[Dict]:
        """Load samples from ChatML JSON file"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single sample and list of samples
        if isinstance(data, dict):
            samples = [data]
        else:
            samples = data
            
        return samples

    def _load_audio_waveform(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load and process audio waveform"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
            return waveform.squeeze(0), self.sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return torch.zeros(1), self.sample_rate

    def _encode_audio_tokens(self, audio_path: str) -> Optional[torch.Tensor]:
        """Encode audio to tokens using audio tokenizer"""
        if not self.audio_tokenizer: 
            return None
        try:
            return self.audio_tokenizer.encode(audio_path)
        except Exception as e:
            logger.error(f"Failed to encode audio {audio_path}: {e}")
            return None

    def _process_chatml_sample(self, sample: Dict) -> tuple:
        """
        Extract reference audio, reference text, and target text from ChatML sample.
        This follows the exact same pattern as the inference script.
        
        Args:
            sample: ChatML format sample dictionary
            
        Returns:
            Tuple of (ref_audio_path, ref_text, target_text, speaker_id)
        """
        try:
            # Parse the ChatML sample structure
            messages = sample["messages"]
            
            # Find user message with reference content
            ref_audio_path = None
            ref_text = None
            target_text = None
            speaker_id = sample.get("speaker", "unknown")
            
            for message in messages:
                if message["role"] == "user":
                    content = message["content"]
                    if isinstance(content, list):
                        # Look for text and audio content
                        text_parts = []
                        for item in content:
                            if item["type"] == "text":
                                text_parts.append(item["text"])
                            elif item["type"] == "audio":
                                if ref_audio_path is None:  # First audio is reference
                                    ref_audio_path = item["audio_url"]
                        
                        if len(text_parts) >= 2:
                            ref_text = text_parts[0]  # First text is reference
                            # Look for target text in the format "Please generate speech for given text..."
                            for text_part in text_parts[1:]:
                                if "Please generate speech" in text_part:
                                    # Extract target text after the instruction
                                    target_text = text_part.split(":")[-1].strip()
                                    break
                            if target_text is None and len(text_parts) > 1:
                                target_text = text_parts[-1]  # Last text as fallback
            
            # Resolve relative paths based on the data file location
            if ref_audio_path and not os.path.isabs(ref_audio_path):
                # Resolve relative to the data file directory
                data_file_dir = os.path.dirname(os.path.abspath(self.data_file))
                resolved_path = os.path.normpath(os.path.join(data_file_dir, ref_audio_path))
                
                # Check if the resolved path exists, if not, try alternative resolutions
                if not os.path.exists(resolved_path):
                    # Try resolving relative to the parent directory
                    parent_dir = os.path.dirname(data_file_dir)
                    alt_resolved_path = os.path.normpath(os.path.join(parent_dir, ref_audio_path))
                    if os.path.exists(alt_resolved_path):
                        resolved_path = alt_resolved_path
                        logger.info(f"Using alternative path resolution: {resolved_path}")
                    else:
                        # Try resolving relative to the grandparent directory
                        grandparent_dir = os.path.dirname(parent_dir)
                        alt_resolved_path = os.path.normpath(os.path.join(grandparent_dir, ref_audio_path))
                        if os.path.exists(alt_resolved_path):
                            resolved_path = alt_resolved_path
                            logger.info(f"Using alternative path resolution: {resolved_path}")
                
                ref_audio_path = resolved_path
            
            if not all([ref_audio_path, ref_text, target_text]):
                logger.warning(f"Missing required components: ref_audio={ref_audio_path}, ref_text={ref_text}, target_text={target_text}")
                return None, None, None, None
                
            return ref_audio_path, ref_text, target_text, speaker_id
            
        except Exception as e:
            logger.error(f"Error processing ChatML sample: {e}")
            return None, None, None, None

    def _prepare_generation_context(self, ref_text: str, ref_audio_path: str) -> tuple:
        """
        Prepare generation context following the inference script pattern.
        This creates the proper message structure for zero-shot voice cloning.
        
        Args:
            ref_text: Reference text
            ref_audio_path: Reference audio path (already resolved)
            
        Returns:
            Tuple of (messages, audio_ids)
        """
        # Create system message
        system_message = Message(
            role="system",
            content="Generate speech in the provided voice."
        )
        
        # Load and encode reference audio
        logger.info(f"Loading reference audio waveform: {ref_audio_path}")
        if not os.path.exists(ref_audio_path):
            logger.error(f"Reference audio file not found: {ref_audio_path}")
            return [], []
            
        try:
            # Load audio waveform
            waveform, sr = torchaudio.load(ref_audio_path)
            logger.info(f"Loaded audio: shape={waveform.shape}, sr={sr}")
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                logger.info(f"Converted to mono: shape={waveform.shape}")
            
            # Encode reference audio for DAC tokens
            audio_tokens = self._encode_audio_tokens(ref_audio_path)
            audio_ids = [audio_tokens] if audio_tokens is not None else []
            if audio_tokens is not None:
                logger.info(f"Audio tokens shape: {audio_tokens.shape}")
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return [], []
        
        # CRITICAL: Follow generation.py pattern for voice prompting
        # Create user message with reference text (no audio token here)
        user_ref_message = Message(
            role="user",
            content=ref_text
        )
        
        # CRITICAL: Assistant responds with AudioContent, not text
        # This is the key difference from the previous broken implementation
        assistant_ref_message = Message(
            role="assistant",
            content=AudioContent(audio_url=ref_audio_path)
        )
        
        messages = [system_message, user_ref_message, assistant_ref_message]
        
        return messages, audio_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        """Get a dataset sample by index"""
        sample = self.samples[idx]
        
        try:
            # Extract components from ChatML sample
            ref_audio_path, ref_text, target_text, speaker_id = self._process_chatml_sample(sample)
            
            if not all([ref_audio_path, ref_text, target_text]):
                logger.warning(f"Skipping sample {idx} due to missing components")
                # Return next sample to avoid interrupting training
                return self.__getitem__((idx + 1) % len(self))
            
            # Prepare generation context following inference script pattern
            messages, audio_ids = self._prepare_generation_context(ref_text, ref_audio_path)
            
            # Add user message with target text
            messages.append(Message(role="user", content=target_text))
            
            # Add assistant message with target audio
            target_audio_path = None
            for message in sample["messages"]:
                if message["role"] == "assistant" and isinstance(message["content"], list):
                    for item in message["content"]:
                        if item["type"] == "audio":
                            target_audio_path = item["audio_url"]
                            break
            
            # Resolve relative paths for target audio with alternative resolution
            if target_audio_path and not os.path.isabs(target_audio_path):
                # Resolve relative to the data file directory
                data_file_dir = os.path.dirname(os.path.abspath(self.data_file))
                resolved_path = os.path.normpath(os.path.join(data_file_dir, target_audio_path))
                
                # Check if the resolved path exists, if not, try alternative resolutions
                if not os.path.exists(resolved_path):
                    # Try resolving relative to the parent directory
                    parent_dir = os.path.dirname(data_file_dir)
                    alt_resolved_path = os.path.normpath(os.path.join(parent_dir, target_audio_path))
                    if os.path.exists(alt_resolved_path):
                        resolved_path = alt_resolved_path
                        logger.info(f"Using alternative path resolution: {resolved_path}")
                    else:
                        # Try resolving relative to the grandparent directory
                        grandparent_dir = os.path.dirname(parent_dir)
                        alt_resolved_path = os.path.normpath(os.path.join(grandparent_dir, target_audio_path))
                        if os.path.exists(alt_resolved_path):
                            resolved_path = alt_resolved_path
                            logger.info(f"Using alternative path resolution: {resolved_path}")
                
                target_audio_path = resolved_path
            
            if target_audio_path:
                messages.append(Message(role="assistant", content=AudioContent(audio_url=target_audio_path)))
            else:
                # Fallback to empty audio content
                messages.append(Message(role="assistant", content=AudioContent(audio_url="")))
            
            chatml_sample = ChatMLSample(messages=messages)
            
            # Process ChatML sample using the same function as inference
            input_tokens, label_tokens, audio_contents, audio_label_contents, _ = prepare_chatml_sample(
                chatml_sample, self.tokenizer
            )

            # Process audio data
            context_audio_tokens = []
            for audio_content in (audio_contents or []):
                if audio_content.audio_url:
                    tokens = self._encode_audio_tokens(audio_content.audio_url)
                    if tokens is not None: 
                        context_audio_tokens.append(tokens)

            label_audio_tokens = []
            for audio_label_content in (audio_label_contents or []):
                if audio_label_content and audio_label_content.audio_url:
                    tokens = self._encode_audio_tokens(audio_label_content.audio_url)
                    if tokens is not None: 
                        label_audio_tokens.append(tokens)

            # Concatenate tensors
            if context_audio_tokens:
                audio_ids_concat = torch.cat(context_audio_tokens, dim=1)
                audio_ids_start = torch.tensor(
                    np.cumsum(np.array([0] + [t.shape[1] for t in context_audio_tokens[:-1]])),
                    dtype=torch.long
                )
            else:
                audio_ids_concat = torch.zeros((8, 0), dtype=torch.long)  # Default to 8 codebooks
                audio_ids_start = torch.tensor([0], dtype=torch.long)

            label_audio_ids = torch.cat(label_audio_tokens, dim=1) if label_audio_tokens else None

            # For ChatMLDatasetSample preparation, also process reference audio for Whisper conditioning
            ref_waveform = None
            ref_sample_rate = None
            # Resolve reference audio path for Whisper conditioning
            if ref_audio_path and not os.path.isabs(ref_audio_path):
                data_file_dir = os.path.dirname(os.path.abspath(self.data_file))
                resolved_ref_audio_path = os.path.normpath(os.path.join(data_file_dir, ref_audio_path))
            else:
                resolved_ref_audio_path = ref_audio_path
                
            if resolved_ref_audio_path and os.path.exists(resolved_ref_audio_path):
                try:
                    ref_waveform, ref_sample_rate = self._load_audio_waveform(resolved_ref_audio_path)
                except Exception as e:
                    logger.warning(f"Could not load reference waveform: {e}")
            
            # Create dataset sample following the exact same pattern as inference
            dataset_sample = ChatMLDatasetSample(
                input_ids=torch.tensor(input_tokens, dtype=torch.long),
                label_ids=torch.tensor(label_tokens, dtype=torch.long),
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                label_audio_ids=label_audio_ids,
                audio_waveforms_concat=ref_waveform if ref_waveform is not None else torch.tensor([]),
                audio_waveforms_start=torch.tensor([0], dtype=torch.long) if ref_waveform is not None else torch.tensor([], dtype=torch.long),
                audio_sample_rate=torch.tensor([ref_sample_rate], dtype=torch.float32) if ref_sample_rate is not None else torch.tensor([], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([0], dtype=torch.long),
            )
            
            return dataset_sample

        except Exception as e:
            logger.error(f"Error processing sample at index {idx}: {e}", exc_info=True)
            # Return next sample to avoid interrupting training
            return self.__getitem__((idx + 1) % len(self))


class HiggsAudioModelWrapper(nn.Module):
    """Wrapper for Higgs Audio v2 model to enable training"""
    def __init__(self, model_path: str, device: str = 'cuda:0', args=None):
        super().__init__()
        if HIGGS_AVAILABLE:
            self.model = HiggsAudioModel.from_pretrained(
                config=HiggsAudioConfig.from_pretrained(model_path),
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch.bfloat16,
                device_map={'': device},
            )
            self.config = self.model.config
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.config = self.model.config
        
        self.model = self.model.to(device)
        
        if args:
            if args.freeze_audio_tower: self.model.freeze_audio_tower()
            if args.freeze_audio_encoder_proj: self.model.freeze_audio_encoder_proj()
            if args.freeze_llm: self.model.freeze_llm()

    @property
    def device(self):
        return self.model.device
          
    def forward(self, **kwargs):
        """
        Forward pass with proper type alignment.
        This follows the same pattern as the inference script.
        """
        # --- Start ultimate fix ---
        # Before data enters the underlying model, forcibly align all floating-point tensor types with the model weights.
        # This is the most reliable method to solve stubborn type mismatch issues.
        model_dtype = next(self.model.parameters()).dtype
        
        for key, value in kwargs.items():
            # Only convert floating-point tensors, ignore integer types (like input_ids)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                kwargs[key] = value.to(model_dtype)
        # --- End ultimate fix ---

        if HIGGS_AVAILABLE:
            return self.model(**kwargs)
        else:
            # Fallback logic also benefits from the above type conversion
            outputs = self.model(input_ids=kwargs.get('input_ids'), attention_mask=kwargs.get('attention_mask'))
            loss = None
            if kwargs.get('label_ids') is not None:
                logits = outputs.logits[..., :-1, :].contiguous()
                labels = kwargs.get('label_ids')[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": outputs.logits}

class HiggsAudioTrainer(Trainer):
    """Custom trainer for Higgs Audio v2 with proper loss computation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = self.model.config
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation"""
        # Type conversion logic has been moved to Model Wrapper, no longer needed here
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss
        
def setup_lora_config(model: nn.Module, lora_config: Dict) -> nn.Module:
    """Setup LoRA configuration for the model following the same pattern as trainer_ddp.py"""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("rank", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        auto_mapping=True
    )
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        model.model.text_model = get_peft_model(model.model.text_model, peft_config)
    elif hasattr(model, 'model'):
        model.model = get_peft_model(model.model, peft_config)
    else:
        model = get_peft_model(model, peft_config)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Higgs Audio v2 for Zero-Shot Voice Cloning (DDP Version)")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to pretrained Higgs Audio model")
    parser.add_argument("--audio_tokenizer_path", type=str, required=True,
                       help="Path to audio tokenizer")
    
    # Data arguments
    parser.add_argument("--train_data_file", type=str, required=True,
                       help="Path to ChatML JSON training file")
    parser.add_argument("--eval_data_file", type=str, default="",
                       help="Path to ChatML JSON evaluation file")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Directory to save model checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                       help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate for optimization")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every X updates steps")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=False,
                       help="Enable LoRA training")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout rate")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for initialization")
    parser.add_argument("--bf16", action="store_true", default=False,
                       help="Enable bfloat16 mixed precision training")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                       choices=["tensorboard", "wandb", "none"],
                       help="Reporting tool")
    parser.add_argument("--logging_dir", type=str, default="./logs",
                       help="Directory for logging")
    
    # Freeze model components
    parser.add_argument("--freeze_audio_tower", action="store_true", default=False,
                       help="Freeze audio tower components")
    parser.add_argument("--freeze_audio_encoder_proj", action="store_true", default=False,
                       help="Freeze audio encoder projection")
    parser.add_argument("--freeze_llm", action="store_true", default=False,
                       help="Freeze LLM components")
    
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    audio_tokenizer = None
    if HIGGS_AVAILABLE:
        try:
            audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path, device=device)
        except Exception as e:
            logger.warning(f"Failed to load audio tokenizer: {e}")

    model = HiggsAudioModelWrapper(args.model_path, device=device, args=args)
    
    if args.bf16:
        model.to(torch.bfloat16)
        logger.info("Model manually cast to bfloat16.")

    if args.use_lora:
        lora_config = {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout}
        model = setup_lora_config(model, lora_config)
        logger.info("LoRA configuration applied")

    train_dataset = ZeroShotVoiceCloningDataset(args.train_data_file, tokenizer, audio_tokenizer)
    eval_dataset = ZeroShotVoiceCloningDataset(args.eval_data_file, tokenizer, audio_tokenizer) if args.eval_data_file else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        fp16=False,
        bf16=args.bf16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=args.report_to,
        logging_dir=args.logging_dir,
        # --- Start ultimate fix ---
        # Set to True to solve DDP hanging issues
        ddp_find_unused_parameters=True,
        # --- End ultimate fix ---
    )

    data_collator = None
    if HIGGS_AVAILABLE and hasattr(model.config, 'audio_in_token_idx'):
        try:
            from transformers import WhisperProcessor
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            data_collator = ExtendedHiggsAudioSampleCollator(
                whisper_processor=whisper_processor,
                audio_in_token_id=model.config.audio_in_token_idx,
                audio_out_token_id=model.config.audio_out_token_idx,
                audio_stream_bos_id=model.config.audio_stream_bos_id,
                audio_stream_eos_id=model.config.audio_stream_eos_id,
                encode_whisper_embed=True,  # Enabled for voice cloning
                pad_token_id=tokenizer.pad_token_id,
                return_audio_in_tokens=False,  # Match inference script
                use_delay_pattern=False,  # Match inference script
                round_to=1,  # Match inference script exactly
                audio_num_codebooks=getattr(model.config, 'audio_num_codebooks', 8),
            )
        except Exception as e:
            logger.warning(f"Failed to setup Higgs collator: {e}. Using fallback.")
    if data_collator is None:
        data_collator = ExtendedHiggsAudioSampleCollator(pad_token_id=tokenizer.pad_token_id)
        logger.warning("Using fallback collator")
        
    trainer = HiggsAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info(f"Starting zero-shot voice cloning training on device: {device}")
    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model()
        logger.info(f"Model saved to {args.output_dir}")
        if args.use_lora:
            lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
            model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            model_to_save.save_pretrained(lora_output_dir)
            logger.info(f"LoRA adapters saved to {lora_output_dir}")

if __name__ == "__main__":
    main()