#!/usr/bin/env python3
"""
Simple test to verify the training_step method is being called correctly
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from trainer.train_v2_ddp import HiggsAudioTrainer
import torch
from transformers import TrainingArguments

# Create a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        
    def forward(self, input_ids=None, **kwargs):
        # Simple forward pass
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 10
        logits = torch.randn(batch_size, seq_len, 10)
        loss = torch.mean(logits)
        return {"loss": loss, "logits": logits}

def test_training_step():
    """Test the training_step method"""
    print("Testing training_step method...")
    
    # Create a simple model
    model = SimpleModel()
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        per_device_train_batch_size=1,
    )
    
    # Create trainer
    trainer = HiggsAudioTrainer(
        model=model,
        args=training_args,
    )
    
    # Create sample inputs
    sample_inputs = {
        "input_ids": torch.randint(0, 10, (1, 10)),
    }
    
    print(f"Sample inputs: {sample_inputs}")
    
    # Call training_step directly
    try:
        loss = trainer.training_step(model, sample_inputs)
        print(f"Training step completed successfully. Loss: {loss}")
        print(f"Loss type: {type(loss)}")
        if hasattr(loss, 'item'):
            print(f"Loss value: {loss.item()}")
    except Exception as e:
        print(f"Error in training_step: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_step()