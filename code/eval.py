import numpy as np
import evaluate
import ray.data as rd
import torch

from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset

from functools import partial
from typing import Union


def compute_metric(eval_pred, metric="f1") -> dict[str, int]:
    metric = evaluate.load(metric)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    return metric.compute(predictions=predictions, references=labels, average="weighted")

def compute_f1_accuracy(eval_pred):
    """
    Compute F1 and accuracy metrics.
    """
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    f1: dict[str, float] = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    accuracy: dict[str, float] = accuracy_metric.compute(predictions=predictions, references=labels)
    
    return {
        "f1": f1["f1"],
        "accuracy": accuracy["accuracy"]
    }


def count_unique_labels(dataset: rd.Dataset, label_column="sentiment"):
    """
    Count unique labels in a Ray dataset.
    """
    unique_labels = set()
    for batch in dataset.iter_batches(batch_size=1000):
        unique_labels.update(batch[label_column])
    return len(unique_labels), unique_labels            
    
    
def eval_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset: Dataset, batch_size, output_dir, num_workers, collate_fn) -> dict[str, Union[float, list[float]]]:
    print("Getting trainer arguments")
    try:
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=num_workers,
            do_train=False,
            do_eval=True,
            logging_strategy="no",
            remove_unused_columns=False
        )
    except Exception as e:
        print("Couldn't get trainer arguments")
        print(e)
    
    print("Instatiating trainer")
    try:
        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=dataset,
            data_collator=partial(collate_fn, tokenizer=tokenizer),
        )
    except Exception as e:
        print("Couldn't instantiate trainer")
        print(e)
    
    # Get predictions
    print("Getting predictions")
    outputs = trainer.predict(dataset)
    y_pred = outputs.predictions.argmax(-1)
    y_true = outputs.label_ids
    
    # Compute metrics
    print("Computing metrics")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "precision": precision,
        "recall": recall,
        "f1": f1 
    }