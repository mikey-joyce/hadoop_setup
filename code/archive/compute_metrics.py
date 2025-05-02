def compute_metrics(eval_pred, metric_name="f1"):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_metric = evaluate.load(metric_name)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    
    return accuracy