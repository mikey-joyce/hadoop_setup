from functools import partial
import re
import time
import sys
import subprocess
import os

from hdfs_utils import clean_empty_parquet_files
from hadoop_setup import setup_hadoop_classpath

import evaluate
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
import raydp
import pyarrow as pa
import pyarrow.dataset as ds
import torch

def parse_rdd(row_str):
    match = re.search(r"Row\(content='(.*?)', sentiment='(.*?)', UID='(.*?)'\)", row_str)
    if match:
        return (match.group(1), match.group(2), match.group(3))
    else:
        return ("", "", "")

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
    
def tokenize_function(examples, tokenizer):
    # print(f"Data keys:\n{examples.keys()}")
    # print(f"Content data type:\n{type(examples['content'][0])}")
    # print(f"Content data :\n{examples['content'][0]}")
    return tokenizer(examples['content'][0], truncation=True)

def compute_metrics(eval_pred) -> dict[str, int]:
      """
    Calculate evaluation metrics based on predictions and labels.

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary of metric names and values
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Load multiple metrics for comprehensive evaluation
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1", "multiclass")
    precision_metric = load_metric("precision", "multiclass")
    recall_metric = load_metric("recall", "multiclass")

    # Calculate each metric
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted", num_classes=len(np.unique(labels)))
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted", num_classes=len(np.unique(labels)))
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted", num_classes=len(np.unique(labels)))

    # Combine all metrics
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"]
    }

def count_unique_labels(dataset, label_column="sentiment"):
    """
    Count unique labels in a Ray dataset.
    """
    unique_labels = set()
    for batch in dataset.iter_batches(batch_size=1000):
        unique_labels.update(batch[label_column])
    return len(unique_labels), unique_labels

def train_func(config):
        """
    Main training function to be executed by Ray.
    """
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    # Get datasets from config
    train_dataset = config["train"]
    eval_dataset = config.get("val")

    # Calculate maximum steps per epoch
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 1)
    max_steps_per_epoch = train_dataset.count() // (batch_size * num_workers)
    print(f"Max steps per epoch: {max_steps_per_epoch}")
    
    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Get label count
    num_labels, unique_labels = count_unique_labels(train_dataset)
    print(f"Detected {num_labels} unique labels: {unique_labels}")
    
    # Load model with correct number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Create data collator using partial
    data_collator = partial(collate_fn, tokenizer=tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        learning_rate=config.get("learning_rate", 2e-5),
        num_train_epochs=config.get("epochs", 3),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=0.1,
        logging_dir="./logs",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="f1" if eval_dataset else None,
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
        eval_dataset=eval_dataset,
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
    if eval_dataset:
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
    n_cpus = int(resources.get("CPU", 1)) - 1
    n_gpus = int(resources.get("GPU", 0))

    return n_cpus, n_gpus

def load_and_prepare_dataset(spark, hdfs_path):
    """
    Reads training data from the given HDFS path using Spark,
    cleans up empty or error parquet files, loads the dataset into Ray,
    verifies the dataset counts, and returns the Ray dataset.
    """
    # Read with Spark
    train_spark_df = spark.read.parquet(hdfs_path)
    train_spark_df.show(5)

    time.sleep(10)
    
    # Clean up Parquet files
    cleanup_results = clean_empty_parquet_files(hdfs_path, spark=spark, verbose=True)
    print(f"Cleaned {cleanup_results['empty_files_removed']} empty files and {cleanup_results['error_files_removed']} error files")

    # Load Ray Dataset using a Hadoop FileSystem instance for pyarrow
    hdfs = pa.fs.HadoopFileSystem("localhost", 9000)

    # Ensure trailing slash for proper reading by Ray
    path = hdfs_path if hdfs_path.endswith("/") else hdfs_path + "/"
    train_dataset = rd.read_parquet(path, filesystem=hdfs)
    print("Dataset type:")
    print(type(train_dataset))
    
    print("Print sample data from Ray Dataset:")
    print(train_dataset.take(2))
    
    # Check if spark dataset to ray dataset worked
    spark_count = train_spark_df.count()
    ray_count = train_dataset.count()
    print(f"Spark DataFrame count: {spark_count}")
    print(f"Ray Dataset count: {ray_count}")

    # Optionally, you could add integrity checks here

    return train_dataset

def main():
    # Setup Hadoop classpath via external function
    setup_hadoop_classpath()

    # Initialize Spark session
    spark = SparkSession.builder.appName("ReadTrain").getOrCreate()

    # Initialize Ray and get available resources
    n_cpus, n_gpus = init_ray()

    # Load, clean, and prepare dataset from HDFS
    hdfs_path = "/phase2/data/train"
    train_dataset = load_and_prepare_dataset(spark, hdfs_path)

    # If you wish to proceed with training, remove or comment out the following sys.exit(0)
    sys.exit(0)

    # Configure resource scaling for Ray Trainer
    worker_resources = {"CPU": round(n_cpus/n_gpus, 0), "GPU": 1} if n_gpus > 0 else {"CPU": n_cpus}
    scaling_config = ScalingConfig(
        num_workers=n_gpus if n_gpus > 0 else 1, 
        use_gpu=bool(n_gpus), 
        resources_per_worker=worker_resources
    )

    # Configure Ray Trainer
    config = {'train': train_dataset}
    trainer = TorchTrainer(train_func, scaling_config=scaling_config, train_loop_config=config)
    result = trainer.fit()
    print("Training completed.")

if __name__ == "__main__":
    main()
