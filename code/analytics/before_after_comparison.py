import os
import torch
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Constants for model paths
ORIGINAL_MODEL_NAME = "roberta-base"
FINETUNED_MODEL_PATH = "/home/ubuntu/ray_results/twitter-roberta-finetune/TorchTrainer_4639e_00000_0_2025-05-02_15-36-54/checkpoint_000004/checkpoint/"

def load_models():
    """
    Load the original and finetuned models along with their tokenizers.
    """
    orig_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME)
    orig_model = AutoModelForSequenceClassification.from_pretrained(ORIGINAL_MODEL_NAME)
    orig_model.eval()
    
    finetuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_PATH)
    finetuned_model.eval()
    
    return (orig_tokenizer, orig_model, finetuned_tokenizer, finetuned_model)

def predict(text, tokenizer, model):
    """
    Tokenizes and performs prediction on the text.
    Returns the predicted class as integer.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return int(torch.argmax(logits, dim=1).item())

def predict_partition(rows):
    """
    Process each partition using a single load of models.
    Yields each row with additional prediction fields.
    """
    # Load models once per partition.
    orig_tokenizer, orig_model, finetuned_tokenizer, finetuned_model = load_models()
    
    for row in rows:
        # Assumes the test dataset has a 'text' column.
        text = row["text"]
        orig_pred = predict(text, orig_tokenizer, orig_model)
        ft_pred = predict(text, finetuned_tokenizer, finetuned_model)
        
        # Append predictions to the row dictionary.
        # Convert pyspark.sql.Row to dict and update.
        row_dict = row.asDict()
        row_dict["orig_pred"] = orig_pred
        row_dict["ft_pred"] = ft_pred
        yield row_dict

def main():
    # Create Spark session
    spark = SparkSession.builder.appName("BeforeAfterComparison").getOrCreate()
    
    # Load the test dataset from HDFS (assumes a distributed parquet dataset)
    test_path = "hdfs:///phase2/data/test/"
    df = spark.read.parquet(test_path)
    
    # Apply predictions using mapPartitions.
    # This returns an RDD where each record now includes 'orig_pred' and 'ft_pred'.
    pred_rdd = df.rdd.mapPartitions(predict_partition)
    
    # Convert the RDD back to a DataFrame (infer schema)
    pred_df = spark.createDataFrame(pred_rdd)
    
    # For comparison, you could compute basic statistics such as the count of matching predictions.
    comparison_df = pred_df.withColumn("match", (col("orig_pred") == col("ft_pred")).cast(IntegerType()))
    
    total = comparison_df.count()
    match_count = comparison_df.groupBy("match").count().filter("match = 1").collect()
    matches = match_count[0]["count"] if match_count else 0
    
    print("Total records: {}".format(total))
    print("Number of matching predictions: {}".format(matches))
    print("Percentage agreement: {:.2f}%".format(100 * matches / total if total > 0 else 0))
    
    spark.stop()

if __name__ == "__main__":
    main()