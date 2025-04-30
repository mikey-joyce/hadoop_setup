"""
This script is a proxy for the finetune.py training function. Instead of redefining
the training logic, it reuses the functions from finetune.py while providing a much
smaller sample dataset and a lighter pretrained model for rapid testing.

Key features:
- Imports the training function (train_func) from finetune.py.
- Creates a dummy dataset (DummyDataset) with minimal samples.
- Uses a small, finetuned model ("distilbert-base-uncased-finetuned-sst-2-english")
  for quick experimentation.
- Constructs a simple configuration to run a single training epoch.
- Enables verification of the syntax, logic, and overall training flow defined in finetune.py.
"""

import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from finetune import train_func  # Now this should work properly

from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self):
        self.samples = [
            {"content": "I love this!", "sentiment": "1"},
            {"content": "I hate this!", "sentiment": "0"},
            {"content": "It is okay.", "sentiment": "1"},
            {"content": "Not my favorite.", "sentiment": "0"},
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def main():
    # Create dummy training and validation datasets
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()

    # Configuration for a quick test run
    config = {
        "train": train_dataset,
        "val": val_dataset,
        "batch_size": 2,
        "num_workers": 1,
        "learning_rate": 1e-5,
        "epochs": 1,
        "weight_decay": 0.0,
        # If additional configurations are expected by finetune.train_func,
        # include them here.
    }
    
    print("Starting test training using functions from finetune.py...")
    train_func(config)
    print("Test training completed.")

if __name__ == "__main__":
    main()