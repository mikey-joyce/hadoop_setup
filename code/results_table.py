import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from typing import List

def metrics_comparison_table(
    y_true_list: List[List[int]],
    y_pred_list: List[List[int]],
    model_names: List[str]
) -> pd.DataFrame:
    """
    Given true/pred lists for each model, returns a DataFrame
    whose rows are metrics (precision, recall, f1) and whose
    columns are the model names.
    """
    # Define which metrics we want
    metrics = ["precision", "recall", "f1"]
    
    # Compute each model’s metrics
    results = {}
    for name, y_true, y_pred in zip(model_names, y_true_list, y_pred_list):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )  # returns (precision, recall, f1, support) cite[1]
        results[name] = {"precision": p, "recall": r, "f1": f1}
    
    # Build DataFrame: rows=metrics, columns=model_names
    df = pd.DataFrame(results, index=metrics)
    return df


if __name__ == "__main__":
    # load predictions
    predictions_path = "/home/ubuntu/hadoop_setup/comparison_results"
    if not os.path.isdir(predictions_path):
        raise FileNotFoundError(f"No such directory {predictions_path}")
    
    df_pretrained = pd.read_parquet(os.path.join(predictions_path, "pretrained_preds.parquet"))
    df_finetuned = pd.read_parquet(os.path.join(predictions_path, "finetuned_preds.parquet"))
    
    y_true_pre = df_pretrained['y_true'].tolist()
    y_pred_pre = df_pretrained['y_pred'].tolist()
    y_true_fine = df_finetuned['y_true'].tolist()
    y_pred_fine = df_finetuned['y_pred'].tolist()
    
    model_names = ["pretrained", "finetuned"]
    df = metrics_comparison_table(
        y_true_list=[y_true_pre, y_true_fine],
        y_pred_list=[y_pred_pre, y_pred_fine],
        model_names=model_names
    )
    
    print(df.to_string())
    
    # save
    save_path = os.path.join(predictions_path, "results.csv")
    df.to_csv(save_path)
