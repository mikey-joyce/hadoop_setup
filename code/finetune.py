import os
import time

from pyspark.sql import SparkSession
import ray
import ray.data as rd
import ray.train.huggingface.transformers
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification


global train        # ensure the training variable has global scope so --> train_func() can access it (we load in data w/ spark before training)
global tokenizer    # I didn't like --> tokenize_function() being within another function, making this global is an easy way to separate them


def tokenize_function(examples):
        return tokenizer(examples["content"], padding="max_length", truncation=True)


def train_func(config):
    transformer = 'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(transformer)
    model = AutoModelForSequenceClassification.from_pretrained(transformer)

    data = (train.select(range(100)).map(tokenize_function, batched=True))
    data.show(5)
    time.sleep(60)


def main():
    # initialize sessions
    spark = SparkSession.builder.appName("ReadTrain").getOrCreate()
    ray.init()
    os.environ["RAY_TRAIN_ENABLE_V2"] = "1"
    time.sleep(5)

    # load in da training data
    rdd = spark.sparkContext.textFile("hdfs:///phase2/data/train")
    train = rd.from_items(list(rdd.toLocalIterator()))
    train.show(5)
    time.sleep(10)

    n_cpus = os.cpu_count()
    scaling_config = ScalingConfig(num_workers=n_cpus, use_gpu=True)

    config = {}
    trainer = TorchTrainer(train_func, scaling_config=scaling_config, train_loop_config=config)
    # result = trainer.fit()


if __name__ == "__main__":
    main()
