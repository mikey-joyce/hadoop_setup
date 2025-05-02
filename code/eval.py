import numpy as np
import evaluate
import ray.data as rd


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
