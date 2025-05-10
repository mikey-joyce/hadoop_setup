# evaluation.py
"""
This script does not use Ray and only uses HF and torch.distributed. 
This is because we use Horovod to distribute the evaluations and
Horovod seamlessly integrates with torch.distribute.

Evaluations:
1. Confusion matrix.
3. Precision, Recall, F1 of pretrained vs Finetuned model
"""

from hadoop_setup import setup_hadoop_classpath
from load_data import load_and_prepare_dataset
from finetune import collate_fn

from pyspark.sql import SparkSession
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader

import pyspark.pandas as ps
import ray.data as rd
import horovod.torch as hvd
import torch

import datetime

PRETRAINED_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
FINETUNED_MODEL_NAME = "zayanhugsAI/twitter_roberta_finetuned_2"


def comparison(model1, model2, dataset: Dataset, tokenizer, batch_size=16, metric=['f1']):
    """Compare model1 and model2 on a dataset"""
    
    # Horovod init
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    
    # Shard dataset per rank
    ds = dataset.remove_columns('UID').shard(num_shards=hvd.size(), index=hvd.rank())
    
    # Prepare dataloader with collate func
    loader = DataLoader(
        dataset=ds,
        batch_size=
    )
    
    

def main():
    spark = None
    
    try:
        setup_hadoop_classpath()
        
        spark = SparkSession.builder.appName("Evaluation").getOrCreate()
        
        # Load dataset
        hdfs_path = "/phase2/data"
        val_path = hdfs_path + "/valid_labels"
        
        try:
            test_data = load_and_prepare_dataset(spark, val_path)
        except:
            print("Couldn't load test data using spark.")
            
        print("Loaded test data.")
        
        # Load models and tokenizer
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)
        finetuned_model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, use_fast=True)

        print("Loaded models and tokenizer")
        
    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
    except Exception as e:
        print("An error occured: ", e)
    finally:
        if spark is not None:
            print("Stopping spark session")
            spark.stop()
            
        print("Exiting program.")
        
        

if __name__ == "__main__":
    main()