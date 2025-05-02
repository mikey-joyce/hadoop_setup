from functools import partial
import re
import time
import sys
import subprocess
import os
import json
import datetime
import logging

from hadoop_setup import setup_hadoop_classpath
from load_data import load_and_prepare_dataset
from eval import compute_metric, compute_f1_accuracy

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
    # outputs["token_type_ids"] = torch.zeros((batch_size, seq_length), dtype=torch.long)

    # Convert potentially float-like strings to float first, then to an integer
    outputs["labels"] = torch.tensor([int(float(label)) for label in batch["sentiment"]])

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
    
    print("Loading model and tokenizer for %s", model_name)
    
    # metric = evaluate.load("accuracy")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    print("Model and tokenizer loaded successfully")
    print("Getting data shards")
    
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("val")
    
    print("Data shards obtained successfully")
    print(train_ds)
    print(val_ds if val_ds else "No validation dataset provided")
    
    if train_ds is None:
        print("Training dataset is None. Please provide a valid dataset.")
        raise ValueError("Training dataset is None. Please provide a valid dataset.")
    if val_ds is None:
        print("No validation dataset provided. Continuing without validation dataset.")
        val_ds = None
        
    collate_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    print("Data collator created successfully")
    
    print("Creating train iterable")
    train_ds_iterable = train_ds.iter_torch_batches(
        batch_size=batch_size,
        collate_fn=collate_with_tokenizer,
    )
    print("Train iterable created successfully")
    print(f"Sample batch: \n{next(iter(train_ds_iterable))}")
    
    print("Creating validation iterable")
    if val_ds is not None:
        val_ds_iterable = val_ds.iter_torch_batches(
            batch_size=batch_size,
            collate_fn=collate_with_tokenizer,
        )
        print("Validation iterable created successfully")
        print(f"Sample validation batch: \n{next(iter(val_ds_iterable))}")
    else:
        val_ds_iterable = None

    print("max steps per epoch %s", max_steps_per_epoch)
    
    print("Obtaining args for Trainer")
    args = TrainingArguments(
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
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
        metric_for_best_model="f1" if val_ds else None,
        greater_is_better=True if val_ds else False,
        fp16=True,
    )
    print("Training arguments obtained successfully")
    
    print("Creating Trainer")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds_iterable,
        eval_dataset=val_ds_iterable,
        tokenizer=tokenizer,
        compute_metrics=compute_f1_accuracy,
    )
    print("Trainer created successfully")
    
    print("Preparing Trainer")
    trainer.add_callback(RayTrainReportCallback)
    trainer = prepare_trainer(trainer)
    print("Trainer prepared successfully")
    
    print("Starting training...")
    trainer.train()
    print("Training finished.")
        
    
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
    spark = None
    ray_initialized = False
    
    try:
        # Setup Hadoop classpath via external function
        setup_hadoop_classpath()

        # Initialize Spark session
        spark = SparkSession.builder.appName("ReadTrain").getOrCreate()

        # Initialize Ray and get available resources
        try:
            n_cpus, n_gpus = init_ray()
            ray_initialized = True
            print("Ray initialized successfully with %d CPUs and %d GPUs", n_cpus, n_gpus)
        except Exception as e:
            print("Ray initialization failed: %s", e)
            sys.exit(1)
            

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
            "epochs": 7,
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
        print("Training completed with result: %s", result)
        
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print("An error occurred: %s", exc_info=True)
    finally:
        if spark is not None:
            print("Stopping Spark session.")
            spark.stop()
        if ray_initialized and ray.is_initialized():
            print("Shutting down Ray.")
            ray.shutdown()
            
        print("Exiting program.")
        
if __name__ == "__main__":
    main()
