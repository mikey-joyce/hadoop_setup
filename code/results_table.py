import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Optional
import numpy as np

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

def metrics_comparison_table_enhanced(
    y_true_list: List[List[int]],
    y_pred_list: List[List[int]],
    model_names: List[str],
    labels: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Given true/pred lists for each model and their names, returns a DataFrame
    whose rows are:
      - one per class: with Support, per-model P/R/F1, and Δ F1 (if two models)
      - Overall Acc., Macro‑F1
    labels: list of class labels to include (in order). If None, inferred via np.unique from all y_true_list.
    """
    n_models = len(model_names)
    assert len(y_true_list) == len(y_pred_list) == n_models

    # Determine classes
    if labels is None:
        # infer from all true labels across models
        all_trues = np.concatenate([y_true for y_true in y_true_list])
        labels = list(np.unique(all_trues))

    supports = None
    per_model = {}

    # Compute per-class P/R/F1 and support
    for name, y_true, y_pred in zip(model_names, y_true_list, y_pred_list):
        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        per_model[name] = {"precision": p, "recall": r, "f1": f1}
        if supports is None:
            supports = sup

    # Build table rows
    rows = []
    for i, cls in enumerate(labels):
        row = {"Class": cls, "Support": int(supports[i])}
        for name in model_names:
            row[f"{name} P"]  = f"{per_model[name]['precision'][i]:.2f}"
            row[f"{name} R"]  = f"{per_model[name]['recall'][i]:.2f}"
            row[f"{name} F₁"] = f"{per_model[name]['f1'][i]:.2f}"
        if n_models == 2:
            delta = per_model[model_names[1]]["f1"][i] - per_model[model_names[0]]["f1"][i]
            row["Δ F₁"] = f"{delta:+.2f}"
        rows.append(row)

    # Compute overall accuracy and macro‑F1
    overall = {
        name: np.mean(np.array(y_pred) == np.array(y_true))
        for name, y_true, y_pred in zip(model_names, y_true_list, y_pred_list)
    }
    macro = {}
    for name, y_true, y_pred in zip(model_names, y_true_list, y_pred_list):
        _, _, mf1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        macro[name] = mf1

    total_support = int(supports.sum())
    # Append aggregate rows
    agg_base = {"Class": "Overall Acc.", "Support": total_support}
    for name in model_names:
        agg_base[f"{name} F₁"] = f"{overall[name]:.2f}"
        agg_base[f"{name} P"] = agg_base[f"{name} R"] = ""
    if n_models == 2:
        agg_base["Δ F₁"] = f"{overall[model_names[1]] - overall[model_names[0]]:+.2f}"
    rows.append(agg_base)

    agg_macro = {"Class": "Macro‑F₁", "Support": ""}
    for name in model_names:
        agg_macro[f"{name} F₁"] = f"{macro[name]:.2f}"
        agg_macro[f"{name} P"] = agg_macro[f"{name} R"] = ""
    if n_models == 2:
        agg_macro["Δ F₁"] = f"{macro[model_names[1]] - macro[model_names[0]]:+.2f}"
    rows.append(agg_macro)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    # load predictions
    predictions_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'comparison_results')
    if not os.path.isdir(predictions_path):
        raise FileNotFoundError(f"No such directory {predictions_path}")
    
    df_pretrained = pd.read_parquet(os.path.join(predictions_path, "pretrained_preds.parquet"))
    df_finetuned = pd.read_parquet(os.path.join(predictions_path, "finetuned_preds.parquet"))
    
    y_true_pre = df_pretrained['y_true'].tolist()
    y_pred_pre = df_pretrained['y_pred'].tolist()
    y_true_fine = df_finetuned['y_true'].tolist()
    y_pred_fine = df_finetuned['y_pred'].tolist()
    
    model_names = ["pretrained", "finetuned"]
    df1 = metrics_comparison_table(
        y_true_list=[y_true_pre, y_true_fine],
        y_pred_list=[y_pred_pre, y_pred_fine],
        model_names=model_names
    )
    
    print(df1.to_string())
    
    # save
    save_path = os.path.join(predictions_path, "results.csv")
    df1.to_csv(save_path, index=False)
    
    df2 = metrics_comparison_table_enhanced(
        y_true_list=[y_true_pre, y_true_fine],
        y_pred_list=[y_pred_pre, y_pred_fine],
        model_names=model_names
    )
    
    # save as csv
    save_path = os.path.join(predictions_path, "results_enhanced.csv")
    df2.to_csv(save_path, index=False)
    
    # save to latex
    latex_str = df2.to_latex(
        index=False,
        escape=False,
        column_format="l r rrr rrr r",
        caption="Per-class comparison with support and ΔF₁",
        label="table: comparison"
    )
    
    print(latex_str)
    latex_file_path = os.path.join(predictions_path, "results_table.tex")
    with open(latex_file_path, "w") as f:
        f.write(latex_str)
    
    print(f"Saved latex table to {latex_file_path}")
