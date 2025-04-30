from functools import partial
import re
import time
import sys
import subprocess
import os
import json

from hadoop_setup import setup_hadoop_classpath
from load_data import load_and_prepare_dataset
from eval import compute_metrics, count_unique_labels

from pyspark.sql import SparkSession
import pyspark.pandas as ps
import ray
import ray.data as rd
import ray.train.huggingface.transformers
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer
import ray.train
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification

import pyarrow as pa
import pyarrow.dataset as ds
import torch
import numpy as np

# def parse_rdd(row_str):
#     match = re.search(r"Row\(content='(.*?)', sentiment='(.*?)', UID='(.*?)'\)", row_str)
#     if match:
#         return (match.group(1), match.group(2), match.group(3))
#     else:
#         return ("", "", "")

def collate_fn(batch, tokenizer):
    """
    Process a batch of data for the model.
    
    Args:
        batch: Dictionary containing the batch data
        tokenizer: HuggingFace tokenizer to use
        
    Returns:
        Dictionary with model inputs and labels
    """

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
    
# def tokenize_function(examples, tokenizer):
#     # print(f"Data keys:\n{examples.keys()}")
#     # print(f"Content data type:\n{type(examples['content'][0])}")
#     # print(f"Content data :\n{examples['content'][0]}")
#     return tokenizer(examples['content'][0], truncation=True)


def train_func(config: dict):
    """
    Main training function to be executed by Ray.
    This function largely follows the train_func from https://docs.ray.io/en/latest/train/examples/transformers/huggingface_text_classification.html#hf-train,
    which takes from https://huggingface.co/docs/transformers/en/training#trainer
    """
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    # Get datasets from config
    train_dataset = config.get("train")
    val_dataset = config.get("val")
    
    if train_dataset is None:
        raise ValueError("Training dataset is None. Please provide a valid dataset.")
    if val_dataset is None:
        choice = input("No validation dataset provided. Restart or continue? (r/c): ")
        if choice.lower() == "r":
            raise ValueError("Validation dataset is None. Please provide a valid dataset.")
        elif choice.lower() == "c":
            print("Continuing without validation dataset.")
        val_dataset = None

    # Calculate maximum steps per epoch
    batch_size = config.get("batch_size", 16)
    learning_rate = config.get("learning_rate", 2e-5)
    epochs = config.get("epochs", 3)
    weight_decay = config.get("weight_decay", 0.01)
    
    # Compute maximum steps per epoch
    num_workers = config.get("num_workers", 1)
    max_steps_per_epoch = train_dataset.count() // (batch_size * num_workers)
    print(f"Max steps per epoch: {max_steps_per_epoch}")
    
    # Load tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Get label count
    num_labels, unique_labels = count_unique_labels(train_dataset)
    print(f"Detected {num_labels} unique labels: {unique_labels}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Create data collator using partial
    data_collator = partial(collate_fn, tokenizer=tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        logging_dir="./logs",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="f1" if val_dataset else None,
        fp16=True,  # Mixed precision training
        push_to_hub=False,
        disable_tqdm=True,  # Cleaner output in distributed environments
        report_to="none",
    )
    
    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics  # Add metrics computation
    )
    
    # Add Ray reporting callback
    trainer.add_callback(RayTrainReportCallback())
    
    # Train the model
    print("Starting training")
    trainer.train()
    
    # Save model and tokenizer
    model_path = "./sentiment_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Return results
    if val_dataset:
        eval_results = trainer.evaluate()
        return eval_results
    return {"status": "completed"}
    
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
    # Setup Hadoop classpath via external function
    setup_hadoop_classpath()

    # Initialize Spark session
    spark = SparkSession.builder.appName("ReadTrain").getOrCreate()

    # Initialize Ray and get available resources
    n_cpus, n_gpus = init_ray()

    # Load, clean, and prepare dataset from HDFS
    hdfs_path = "/phase2/data"
    train_path = f"{hdfs_path}/train"
    val_path = f"{hdfs_path}/valid_labels"
    
    train_dataset = load_and_prepare_dataset(spark, train_path)
    val_dataset = load_and_prepare_dataset(spark, val_path)

    # Determine per-worker resource allocation
    if n_gpus > 0:
        if n_gpus > n_cpus:
            cpus_per_worker = 1  # Ensure each GPU worker gets at least 1 CPU
        else:
            cpus_per_worker = n_cpus // n_gpus  # Use integer division for clarity
        worker_resources = {"CPU": cpus_per_worker, "GPU": 1}
    else:
        worker_resources = {"CPU": n_cpus}

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=n_gpus if n_gpus > 0 else 1,
        use_gpu=bool(n_gpus),
        resources_per_worker=worker_resources
    )
    
    # Hyperparameters
    hyperparameters = {
        "batch_size": 16,
        "num_workers": n_gpus if n_gpus > 0 else 1,
        "learning_rate": 2e-5,
        "epochs": 3,
        "weight_decay": 0.01
    }

    # Configure Ray Trainer
    config = {
        'train': train_dataset, 
        'val': val_dataset,
        **hyperparameters
    }
    
    # Display training configuration including per-worker resources before training starts
    display_training_config(hyperparameters, scaling_config)

    trainer = TorchTrainer(
        train_func, 
        scaling_config=scaling_config, 
        train_loop_config=config,
    )
    
    result = trainer.fit()
    print("Training completed with result: ", result)

if __name__ == "__main__":
    main()
