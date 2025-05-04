from functools import partial
import re
import time
import sys
import subprocess
import os
import json
import datetime
import argparse

from hadoop_setup import setup_hadoop_classpath
from load_data import load_and_prepare_dataset
from eval import compute_metric, compute_f1_accuracy
from finetune import collate_fn, train_func, init_ray, display_training_config

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

MODEL_NAME = "zayanhugsAI/twitter_roberta_finetuned"
NUM_LABELS = 3

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            
def main():
    parser = argparse.ArgumentParser(description="Ray Train Finetune Script")
    parser.add_argument("-gs", "--global_shuffle", action="store_true", help="Enable global shuffle")
    args = parser.parse_args()
    
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
        
        if args.global_shuffle:
            print("Global shuffle enabled. Shuffling datasets...")
            train_dataset = train_dataset.random_shuffle()
            val_dataset = val_dataset.random_shuffle()

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
            'name': MODEL_NAME,
            'model_name': MODEL_NAME,
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
                    checkpoint_score_attribute="eval_f1",
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
