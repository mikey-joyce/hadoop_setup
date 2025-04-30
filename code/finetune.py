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
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import raydp
import pyarrow as pa
import pyarrow.dataset as ds

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

def train_func(config):
    transformer = 'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(transformer, use_fast=True) # use_fast may not be available. But if it is, use it.
    model = AutoModelForSequenceClassification.from_pretrained(transformer)
    metric = evaluate.load("accuracy")

    token_func = partial(tokenize_function, tokenizer=tokenizer)
    data = config["train"].map_batches(token_func, batch_size=100, batch_format="numpy")
    print("Hello?")
    # data.show(5)
    # time.sleep(60)
    # data.show(5)
    # time.sleep(60)
    preview = data.take_batch(5)
    print(f"Ray Data Preview: \n{preview}")

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
