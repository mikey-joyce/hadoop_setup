from functools import partial
import re
import time
import sys
import subprocess
import os
import json
import datetime

from hadoop_setup import setup_hadoop_classpath
from load_data import load_and_prepare_dataset
from eval import compute_metrics, count_unique_labels

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

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
NUM_LABELS = 3

# def parse_rdd(row_str):
#     match = re.search(r"Row\(content='(.*?)', sentiment='(.*?)', UID='(.*?)'\)", row_str)
#     if match:
#         return (match.group(1), match.group(2), match.group(3))
#     else:
#         return ("", "", "")

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

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
        raise ValueError("Tokenizer is not provided to collate function")

    outputs = tokenizer(
            list(batch["content"]),
            truncation=True,
            padding="longest",
            max_length=512,
            return_tensors="pt",
            )
    

    seq_length = outputs["input_ids"].shape[1]
    batch_size = outputs["input_ids"].shape[0]
    
    # create token_type_ids and labels
    outputs["token_type_ids"] = torch.zeros((batch_size, seq_length), dtype=torch.long)

    # Convert potentially float-like strings to float first, then to an integer
    outputs["labels"] = torch.tensor([int(float(label)) for label in batch["sentiment"]])

    return outputs
    
# def tokenize_function(examples, tokenizer):
#     # print(f"Data keys:\n{examples.keys()}")
#     # print(f"Content data type:\n{type(examples['content'][0])}")
#     # print(f"Content data :\n{examples['content'][0]}")
#     return tokenizer(examples['content'][0], truncation=True)

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
        print("No validation dataset provided. Continuing without validation dataset.")
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
        disable_tqdm=True,
        no_cuda=not use_gpu,
        report_to="none",
        output_dir=f"./results/{name}_{current_time}",
        logging_dir=f"./logs/{name}_{current_time}",
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="accuracy" if val_ds else None,
        greater_is_better=True if val_ds else False,
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
    n_cpus = int(resources.get("CPU", 1))
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
        num_workers = n_gpus
        cpus_per_worker = 1
        worker_resources = {"CPU": cpus_per_worker, "GPU": 1}
    else:
        num_workers = 1
        cpus_per_worker = n_cpus
        worker_resources = {"CPU": cpus_per_worker}

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(n_gpus),
        resources_per_worker=worker_resources,
    )
    
    batch_size = 16
    num_workers = scaling_config.num_workers
    
    # Hyperparameters
    hyperparameters = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "learning_rate": 2e-5,
        "epochs": 3,
        "max_steps_per_epoch": train_dataset.count() // (batch_size * num_workers),  # Adjust as needed
        "weight_decay": 0.01
    }

    # Configure Ray Trainer
    config = {
        'name': "twitter-roberta-finetune",
        'model_name': "cardiffnlp/twitter-roberta-base-sentiment",
        'num_labels': 3,
        **hyperparameters
    }
    
    # Display training configuration including per-worker resources before training starts
    display_training_config(hyperparameters, scaling_config)

    trainer = TorchTrainer(
        train_func, 
        scaling_config=scaling_config, 
        datasets={"train": train_dataset, "val": val_dataset},
        train_loop_config=config,
        run_config=RunConfig(
            name=config["name"],
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="accuracy",
                checkpoint_score_order="max",
            )
        )
    )
    
    result = trainer.fit()
    print("Training completed with result: ", result)
    
    ray.shutdown()

if __name__ == "__main__":
    main()
