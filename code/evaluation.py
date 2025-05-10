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
from eval import eval_model

from pyspark.sql import SparkSession
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader

import pyspark.pandas as ps
import ray.data as rd
import horovod.torch as hvd
import torch

import datetime
import multiprocessing as mp
from functools import partial

PRETRAINED_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
FINETUNED_MODEL_NAME = "zayanhugsAI/twitter_roberta_finetuned_2"

def detect_hardware():
    n_cpus = mp.cpu_count()
    n_gpus = torch.cuda.device_count()
    
    print(f"[Hardware] logical cpus available: {n_cpus}")
    print(f"[Hardware] CUDA gpus available: {n_gpus}")

    return n_cpus, n_gpus

def get_num_workers(reserve_cores=1):
    n_cpus, n_gpus = detect_hardware()
    usable = max(1, n_cpus-reserve_cores)
    per_rank = usable // max(1, n_gpus)
    num_workers = max(1, per_rank)
    
    print(f"[Workers] reserve {reserve_cores} core(s) => usable cpus: {usable}")
    print(f"[Workers] distributing across {max(1, n_gpus)} rank(s) => "
          f"num_workers per rank: {num_workers}")
    
    return num_workers

def comparison(model1, model2, dataset: Dataset, tokenizer, batch_size=16, metrics=['f1'], num_workers=1):
    """Compare model1 and model2 on a dataset"""
    
    # Horovod init
    try:
        print("Initializing Horovod")
        hvd.init()
    except Exception as e:
        print("Couldn't initialize Horovd. Exception:")
        print(e)
    
    device = torch.device(f"cuda:{hvd.local_rank()}" if torch.cuda.is_available() else "cpu")    
    torch.cuda.set_device(device)
    
    # Shard dataset per rank
    print("Getting data shard")
    try:
        ds = dataset.remove_columns('UID').shard(num_shards=hvd.size(), index=hvd.rank())
    except Exception as e:
        print("Couldn't load data shard.")
        print(e)
    
    try:
        print("Creating data loader")
        # Prepare dataloader with collate func
        loader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, tokenizer)
        )
    except Exception as e:
        print("Couldn't instantiate data loader.")
        print(e)
    
    model1_scores = eval_model(model1, loader, metrics, device)
    model2_scores = eval_model(model2, loader, metrics, device)
    
    return model1_scores, model2_scores
    

def main():
    spark = None
    
    try:
        setup_hadoop_classpath()
        spark = SparkSession.builder.appName("Evaluation").getOrCreate()
        
        # Set num of workers
        num_workers = get_num_workers()
        
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