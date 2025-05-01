"""
This script tests the entire training pipeline defined in finetune.py.
It mimics the logic in finetune.py's main() function by creating dummy Ray datasets,
setting up a dummy scaling configuration, and launching a TorchTrainer to run a brief training run.
The dummy data uses three sentiment labels:
    0 -> Negative sentiment
    1 -> Neutral sentiment
    2 -> Positive sentiment
"""

import sys
import os
from ray.data import from_items
import ray
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

# Add the project root to sys.path so that finetune.py can be imported correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from finetune import train_func, display_training_config

def main():
    # Create dummy training and validation Ray datasets using a list of dictionaries.
    data = [
        {"content": "I hate this!", "sentiment": "0"},       # Negative sentiment
        {"content": "It is okay.", "sentiment": "1"},          # Neutral sentiment
        {"content": "I love this!", "sentiment": "2"},         # Positive sentiment
        {"content": "Not my favorite.", "sentiment": "0"},       # Negative sentiment
        {"content": "Could be better.", "sentiment": "1"},       # Neutral sentiment
        {"content": "Absolutely fantastic!", "sentiment": "2"},  # Positive sentiment
    ]
    train_dataset = from_items(data)
    val_dataset = from_items(data)
    
    # Define hyperparameters (using the same names as in finetune.py)
    batch_size = 2
    num_workers = 1
    hyperparameters = {
        "batch_size": batch_size,
        "epochs": 1,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        # Set max_steps_per_epoch based on the dummy dataset's count
        "max_steps_per_epoch": train_dataset.count() // (batch_size * num_workers)
    }
    
    # Prepare scaling configuration for TorchTrainer (dummy values for testing)
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=False,
        resources_per_worker={"CPU": 1}
    )
    
    # Display the training configuration for inspection
    display_training_config(hyperparameters, scaling_config)
    
    # Prepare the training configuration for the training function in finetune.py
    config = {
        'name': "twitter-roberta-finetune-test",
        'model_name': "cardiffnlp/twitter-roberta-base-sentiment",
        'num_labels': 3,
        **hyperparameters,
        "use_gpu": False
    }
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Configure TorchTrainer to run the training pipeline from finetune.py
    trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        datasets={"train": train_dataset, "eval": val_dataset},
        train_loop_config=config,
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="accuracy",
                checkpoint_score_order="max",
            )
        )
    )
    
    print("Starting test training using TorchTrainer and functions from finetune.py...")
    result = trainer.fit()
    print("Test training completed with result:", result)
    
    ray.shutdown()

if __name__ == "__main__":
    main()