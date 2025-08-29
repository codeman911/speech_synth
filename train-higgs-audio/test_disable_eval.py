#!/usr/bin/env python3
"""
Test script to verify that evaluation is properly disabled when --disable_eval is set
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the trainer directory to the path so we can import train_v2_ddp
sys.path.append(os.path.join(os.path.dirname(__file__), 'trainer'))

class TestDisableEval(unittest.TestCase):
    
    @patch('train_v2_ddp.argparse.ArgumentParser.parse_args')
    @patch('train_v2_ddp.HiggsAudioModelWrapper')
    @patch('train_v2_ddp.AutoTokenizer')
    @patch('train_v2_ddp.ZeroShotVoiceCloningDataset')
    @patch('train_v2_ddp.HiggsAudioTrainer')
    def test_evaluation_disabled(self, mock_trainer, mock_dataset, mock_tokenizer, mock_model, mock_parse_args):
        """Test that evaluation is disabled when --disable_eval is set"""
        # Mock the arguments with disable_eval=True
        mock_args = MagicMock()
        mock_args.disable_eval = True
        mock_args.use_lora = False
        mock_args.model_path = "test_model"
        mock_args.audio_tokenizer_path = "test_tokenizer"
        mock_args.train_data_file = "test_data.json"
        mock_args.eval_data_file = ""
        mock_args.validation_split = 0.0
        mock_args.output_dir = "./test_output"
        mock_args.num_train_epochs = 1
        mock_args.per_device_train_batch_size = 1
        mock_args.per_device_eval_batch_size = 1
        mock_args.learning_rate = 5e-5
        mock_args.warmup_steps = 10
        mock_args.logging_steps = 10
        mock_args.save_steps = 100
        mock_args.eval_steps = 50
        mock_args.seed = 42
        mock_args.bf16 = False
        mock_args.report_to = "tensorboard"
        mock_args.logging_dir = "./test_logs"
        mock_args.freeze_audio_tower = False
        mock_args.freeze_audio_encoder_proj = False
        mock_args.freeze_llm = False
        
        mock_parse_args.return_value = mock_args
        
        # Mock other dependencies
        mock_model.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_dataset.return_value = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Import and run main function
        import train_v2_ddp
        with patch.object(train_v2_ddp, '__name__', '__main__'):
            train_v2_ddp.main()
        
        # Verify that the trainer was initialized with disable_eval=True
        mock_trainer.assert_called_once()
        call_kwargs = mock_trainer.call_args[1]
        self.assertEqual(call_kwargs.get('disable_eval'), True)
        
        # Verify that training arguments have evaluation disabled
        training_args = call_kwargs.get('args')
        self.assertEqual(training_args.evaluation_strategy, "no")
        self.assertIsNone(training_args.eval_steps)
        self.assertEqual(training_args.load_best_model_at_end, False)
        self.assertIsNone(training_args.metric_for_best_model)

if __name__ == '__main__':
    unittest.main()