from functools import partial
import logging
import time

import evaluate
from pyspark.sql import SparkSession
import ray
import ray.data as rd
import ray.train.huggingface.transformers
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification


# setup custom logger for debugging purposes
logger = logging.getLogger("finetune_logger")
handler = logging.FileHandler("finetune_log.txt")
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def tokenize_function(examples, tokenizer):
        logger.info(f"Examples preview:\n{examples}")
        examples.show(5)
        return tokenizer(examples["content"], padding="max_length", truncation=True)


def train_func(config):
    transformer = 'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(transformer)
    model = AutoModelForSequenceClassification.from_pretrained(transformer)
    metric = evaluate.load("accuracy")

    token_func = partial(tokenize_function, tokenizer=tokenizer)
    data = (config["train"].map_batches(token_func, batch_size=100, batch_format="numpy"))
    # data.show(5)
    # time.sleep(60)
    preview = data.take(5)
    logger.info(f"Ray Data Preview: \n{preview}")


def main():
    # initialize sessions
    spark = SparkSession.builder.appName("ReadTrain").getOrCreate()
    ray.init()
    time.sleep(5)

    # load in da training data
    rdd = spark.sparkContext.textFile("hdfs:///phase2/data/train")
    train = rd.from_items(list(rdd.toLocalIterator()))
    train.show(5)
    time.sleep(10)

    resources = ray.cluster_resources()
    n_cpus = int(resources.get("CPU", 1)) - 1
    n_gpus = int(resources.get("GPU", 0))
    scaling_config = ScalingConfig(num_workers=n_gpus, use_gpu=True, resources_per_worker={"CPU": round(n_cpus/n_gpus, 0), "GPU":1})

    config = {'train': train}
    trainer = TorchTrainer(train_func, scaling_config=scaling_config, train_loop_config=config)
    result = trainer.fit()


if __name__ == "__main__":
    main()
