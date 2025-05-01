"""
This script tests the entire training pipeline defined in finetune.py.
It mimics the logic in finetune.py's main() function by creating dummy Ray datasets,
setting up a dummy scaling configuration, and launching a TorchTrainer to run a brief training run.
The dummy data uses three sentiment labels:
    0 -> Negative sentiment
    1 -> Neutral sentiment
    2 -> Positive sentiment
"""
from functools import partial
import re
import time
import sys
import subprocess
import os
import json
import datetime

from pyspark.sql import SparkSession
import pyspark.pandas as ps
import ray
import ray.data as rd
import ray.train.huggingface.transformers
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
import ray.train
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import evaluate
import pyarrow as pa
import pyarrow.dataset as ds
import torch
import numpy as np

# MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
# TOKENIZER = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
# BATCH_SIZE = 16
# NUM_LABELS = 3

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")

def compute_metrics(eval_pred, metric_name="accuracy"):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_metric = evaluate.load(metric_name)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    
    return accuracy

def collate_fn(batch, tokenizer):
    """
    Process a batch of data for the model.
    
    Args:
        batch: Dictionary containing the batch data
        tokenizer: HuggingFace tokenizer to use
        
    Returns:
        Dictionary with model inputs and labels
    """
    
    if tokenizer is None:
        print("Tokenizer is not provided. Use default? (default = cardiffnlp/twitter-roberta-base-sentiment)")
        choice = input("Enter 'y' to use default, 'n' to exit: ")
        if choice.lower() == 'y':
            tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", use_fast=True)
        elif choice.lower() == 'n':
            print("Exiting...")
            sys.exit(1)

    outputs = tokenizer(
            list(batch["content"]),
            truncation=True,
            padding="longest",
            return_tensors="pt",
            )

    outputs["labels"] = torch.tensor([int(label) for label in batch["sentiment"]])

    if torch.cuda.is_available():
        outputs = {k: v.cuda() for k, v in outputs.items()}

    return outputs

def train_func(config):
    use_gpu = True if torch.cuda.is_available() else False
    print(f"Is CUDA available: {use_gpu}")
    
    name = config.get("name", "twitter-roberta-finetune")
    model_name = config.get("model_name", "cardiffnlp/twitter-roberta-base-sentiment")
    num_labels = config.get("num_labels", 3)
    max_steps_per_epoch = config.get("max_steps_per_epoch", 1000)
    batch_size = config.get("batch_size", 16)
    learning_rate = config.get("learning_rate", 2e-5)
    epochs = config.get("epochs", 10)
    weight_decay = config.get("weight_decay", 0.01)
    use_gpu = config.get("use_gpu", False)
    
    metric = evaluate.load("accuracy")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("val")
    
    if train_ds is None:
        raise ValueError("Training dataset is None. Please provide a valid dataset.")
    if val_ds is None:
        choice = input("No validation dataset provided. Restart or continue? (r/c): ")
        if choice.lower() == "r":
            raise ValueError("Validation dataset is None. Please provide a valid dataset.")
        elif choice.lower() == "c":
            print("Continuing without validation dataset.")
        val_ds = None
        
    collate_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)

    train_ds_iterable = train_ds.iter_torch_batches(
        batch_size=batch_size,
        collate_fn=collate_with_tokenizer,
    )
    
    if val_ds is not None:
        val_ds_iterable = val_ds.iter_torch_batches(
            batch_size=batch_size,
            collate_fn=collate_with_tokenizer,
        )
    else:
        val_ds_iterable = None

    print("max steps per epoch ", max_steps_per_epoch)
    
    args = TrainingArguments(
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        push_to_hub=False,
        max_steps=max_steps_per_epoch * epochs,
        disable_tqdm=False,
        no_cuda=not use_gpu,
        report_to="none",
        output_dir=f"./results/{name}_{current_time}",
        logging_dir=f"./logs/{name}_{current_time}",
        load_best_model_at_end=True if val_ds else False,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds_iterable,
        eval_dataset=val_ds_iterable,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.add_callback(RayTrainReportCallback)
    trainer = prepare_trainer(trainer)
    
    print("Starting training...")
    trainer.train()


def init_ray():
    """
    Initializes Ray and returns available CPU and GPU counts.
    """
    ray.init()
    assert ray.is_initialized(), "Ray initialization failed."

    time.sleep(5)

    resources = ray.cluster_resources()
    n_cpus = max(int(resources.get("CPU", 1)) - 1, 1) # Subtract 1 to leave one CPU for driver
    n_gpus = int(resources.get("GPU", 0))

    return n_cpus, n_gpus

def display_training_config(training_config: dict, scaling_config: ScalingConfig):
    """
    Displays the training and scaling configuration in a friendly format.
    """
    print("=== Training Configuration ===")
    for key, value in training_config.items():
        print(f"{key}: {value}")
    print("==============================")
    print("=== Scaling Configuration ===")
    print(f"num_workers: {scaling_config.num_workers}")
    print(f"use_gpu: {scaling_config.use_gpu}")
    print("resources_per_worker:")
    print(json.dumps(scaling_config.resources_per_worker, indent=4))
    print("==============================")



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
    train_dataset = rd.from_items(data)
    val_dataset = rd.from_items(data)
    
    # Define hyperparameters (using the same names as in finetune.py)
    batch_size = 2
    num_workers = 1
    hyperparameters = {
        "batch_size": batch_size,
        "epochs": 3,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
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
        datasets={"train": train_dataset, "val": val_dataset},
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