from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
import os

def plot_cm(
    y_true: List[int],
    y_pred: List[int],
    labels: List[int],
    title: str
) -> plt.Figure:
    """
    Build and return a confusion‑matrix Figure using sklearn’s helper.
    """
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        normalize="true",
        cmap=plt.cm.Blues,
        ax=ax,
        colorbar=True
    )
    ax.set_title(title)
    return fig

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
    
    labels = [0,1,2]
    fig_pre = plot_cm(y_true=y_true_pre, y_pred=y_pred_pre, labels=labels, title="Confusion matrix for Pretrained model")
    fig_fine = plot_cm(y_true=y_true_fine, y_pred=y_pred_fine, labels=labels, title="Confusion matrix for finetuned model")
    
    # save figures
    save_path = predictions_path
    fig_pre.savefig(os.path.join(save_path, "pretrained_cm.png"))
    fig_fine.savefig(os.path.join(save_path, "finetuned_cm.png"))
    
    plt.close(fig_pre)
    plt.close(fig_fine)