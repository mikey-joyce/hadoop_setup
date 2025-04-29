from functools import partial
import re
import time

import evaluate
from pyspark.sql import SparkSession
import pyspark.pandas as ps
import ray
import ray.data as rd
import ray.train.huggingface.transformers
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification


def parse_rdd(row_str):
    match = re.search(r"Row\(content='(.*?)', sentiment='(.*?)', UID='(.*?)'\)", row_str)
    if match:
        return (match.group(1), match.group(2), match.group(3))
    else:
        return ("", "", "")


def tokenize_function(examples, tokenizer):
        # print(f"Data keys:\n{examples.keys()}")
        # print(f"Content data type:\n{type(examples['content'][0])}")
        # print(f"Content data :\n{examples['content'][0]}")
        return tokenizer(examples['content'][0], truncation=True)


def train_func(config):
    transformer = 'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(transformer)
    model = AutoModelForSequenceClassification.from_pretrained(transformer)
    metric = evaluate.load("accuracy")

    token_func = partial(tokenize_function, tokenizer=tokenizer)
    data = (config["train"].map_batches(token_func, batch_size=100, batch_format="numpy"))
    print("Hello?")
    # data.show(5)
    # time.sleep(60)
    preview = data.take_batch(5)
    print(f"Ray Data Preview: \n{preview}")


def main():
    # initialize sessions
    spark = SparkSession.builder.appName("ReadTrain").getOrCreate()
    ray.init()
    time.sleep(5)

    # load in training data from Parquet (created in parse_data.py)
    train_spark_df = spark.read.parquet("hdfs:///phase2/data/train")
    train_spark_df.show(5)
    time.sleep(10)

    # convert Spark DataFrame to a Ray Dataset
    train_dataset = rd.from_spark(train_spark_df)

    resources = ray.cluster_resources()
    n_cpus = int(resources.get("CPU", 1)) - 1
    n_gpus = int(resources.get("GPU", 0))
    scaling_config = ScalingConfig(
        num_workers=n_gpus, 
        use_gpu=True, 
        resources_per_worker={"CPU": round(n_cpus/n_gpus, 0), "GPU": 1}
    )

    config = {'train': train_dataset}
    trainer = TorchTrainer(train_func, scaling_config=scaling_config, train_loop_config=config)
    result = trainer.fit()


if __name__ == "__main__":
    main()
