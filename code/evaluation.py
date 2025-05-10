# evaluation.py
"""
This script does not use Ray and only uses HF and torch.distributed. 

Evaluations:
1. Confusion matrix.
3. Precision, Recall, F1 of pretrained vs Finetuned model
"""

from eval import eval_model
from hadoop_setup import setup_hadoop_classpath

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, load_dataset, logging

import torch
import multiprocessing as mp
import evaluate
import matplotlib.pyplot as plt
import os

logging.disable_progress_bar()
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

def collate_fn(
    batch: list[dict],
    tokenizer,
    max_length: int = 512
) -> dict:
    """
    batch: List of examples, each a dict with keys 'content' and 'sentiment'.
    tokenizer: HuggingFace tokenizer.
    """
    # 1) Extract lists of texts and labels
    texts  = [example["content"]   for example in batch]
    labels = [example["sentiment"] for example in batch]

    # 2) Tokenize the texts
    outputs = tokenizer(
        texts,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt"
    )

    # 3) Convert labels to longs
    outputs["labels"] = torch.tensor(
        [int(float(lbl)) for lbl in labels], 
        dtype=torch.long
    )

    return outputs

def comparison(pretrained_model, finetuned_model, dataset: Dataset, tokenizer, output_dir, collate_fn, batch_size=16, num_workers=1):
    """Compare model1 and model2 on a dataset"""
    
    results_pretrained = eval_model(
        model=pretrained_model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=batch_size,
        output_dir=output_dir + "/pretrained",
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    results_finetuned = eval_model(
        model=finetuned_model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=batch_size,
        output_dir=output_dir + "/finetuned",
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return results_pretrained, results_finetuned

def plot_cm(cm_output, title: str = "Confusion Matrix") -> plt.Figure:
    """
    Plots an N×N confusion matrix from HF Evaluate's output.
    cm_output: {"confusion_matrix": List[List[int]], "labels": List[int]}
    """
    cm = cm_output["confusion_matrix"]
    labels = cm_output.get('labels', list(range(len(cm))))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    
    return fig    
    

def main():
    print("Running evaluation script.")
    
    setup_hadoop_classpath()
    
    # Set num of workers
    num_workers = get_num_workers()
    
    # Load dataset
    hdfs_path = "hdfs:///phase2/data"
    val_path = hdfs_path + "/valid_labels"
    print(f"Using data from {val_path} to evaluate.")
    try:
        validation_datasetdict = load_dataset(
            "parquet",
            data_files={"valid_labels": f"{val_path}/*.parquet"}
        )
        validation_data = validation_datasetdict['valid_labels']
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {val_path}") from e
        
    print("Loaded test data.")
    
    # Remove "UID column"
    validation_data = validation_data.remove_columns("UID")
    
    # Load models and tokenizer
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, use_fast=True)

    print("Loaded models and tokenizer")
    
    # Compare
    output_dir = "/home/ubuntu/hadoop_setup/comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    results_pretrained, results_finetuned = comparison(
        pretrained_model=pretrained_model,
        finetuned_model=finetuned_model,
        dataset=validation_data,
        tokenizer=tokenizer,
        output_dir=output_dir,
        collate_fn=collate_fn,
        num_workers=num_workers,
        batch_size=64
    )
    
    # plot confusion matrix
    cm_metric = evaluate.load("confusion_matrix")   
    cm_results_pretrained = cm_metric.compute(predictions=results_pretrained['y_pred'], references=results_pretrained['y_true'], labels=[0,1,2])
    cm_results_finetuned= cm_metric.compute(predictions=results_finetuned['y_pred'], references=results_finetuned['y_true'], labels=[0,1,2])
    
    cm_pretrained = plot_cm(cm_results_pretrained, "Pretrained model CM")
    cm_finetuned = plot_cm(cm_results_finetuned, "Finetuned model CM")
    
    # Save confusion matrix
    cm_pretrained.savefig(output_dir + "/pretrained.png")
    cm_finetuned.savefig(output_dir + "/finetuned.png")
    
    plt.close(cm_pretrained)
    plt.close(cm_finetuned)

if __name__ == "__main__":
    main()
