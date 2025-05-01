import numpy as np
import evaluate
import ray.data as rd

def compute_metrics(eval_pred, metric_name="accuracy"):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_metric = evaluate.load(metric_name)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    
    return accuracy

def compute_multiple_metrics(eval_pred) -> dict[str, int]:
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
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1", "multiclass")
    precision_metric = evaluate.load("precision", "multiclass")
    recall_metric = evaluate.load("recall", "multiclass")

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

def count_unique_labels(dataset: rd.Dataset, label_column="sentiment"):
    """
    Count unique labels in a Ray dataset.
    """
    unique_labels = set()
    for batch in dataset.iter_batches(batch_size=1000):
        unique_labels.update(batch[label_column])
    return len(unique_labels), unique_labels
