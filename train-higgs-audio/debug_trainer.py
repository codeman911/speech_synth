#!/usr/bin/env python3
"""
Debug script to verify data flow in HiggsAudioTrainer
"""

import os
import sys
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments
from trainer.train_v2_ddp import HiggsAudioTrainer, ExtendedHiggsAudioBatchInput

class DummyDataset(Dataset):
    """Dummy dataset for testing"""
    def __init__(self):
        self.data = [i for i in range(10)]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Return a simple batch input
        return ExtendedHiggsAudioBatchInput(
            input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            attention_mask=torch.ones((1, 5)),
            label_ids=torch.tensor([[1, 2, 3, 4, 5]])
        )

class DummyModel(torch.nn.Module):
    """Dummy model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        
    def forward(self, input_ids=None, attention_mask=None, label_ids=None, **kwargs):
        # Simple forward pass that returns a loss
        logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 100)  # Simulate logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 100), 
            label_ids.view(-1), 
            ignore_index=-100
        )
        return {"loss": loss, "logits": logits}

def test_training_step():
    """Test the training_step method"""
    print("Testing HiggsAudioTrainer training_step method...")
    
    # Create dummy model and dataset
    model = DummyModel()
    dataset = DummyDataset()
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        per_device_train_batch_size=1,
        logging_steps=1,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = HiggsAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Get a sample input
    sample_input = dataset[0]
    print(f"Sample input type: {type(sample_input)}")
    print(f"Sample input keys: {dir(sample_input)}")
    
    # Test training_step method directly
    print("\nTesting training_step method directly...")
    try:
        # Call training_step with the sample input
        loss = trainer.training_step(model, sample_input)
        print(f"Training step completed successfully. Loss: {loss}")
    except Exception as e:
        print(f"Error in training_step: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nTest completed.")

if __name__ == "__main__":
    test_training_step()