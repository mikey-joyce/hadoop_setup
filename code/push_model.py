from transformers import AutoModelForSequenceClassification
from ray.train import Checkpoint

    
def main():
    checkpoint_path = "/home/ubuntu/ray_results/twitter-roberta-finetune/TorchTrainer_4639e_00000_0_2025-05-02_15-36-54/checkpoint_000005/checkpoint/"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    
    print("Model loaded successfully.")
    
    model.push_to_hub(commit_message="Trained for 4 epochs")
    
    print("Model pushed to Hugging Face Hub successfully.")
    
if __name__ == "__main__":
    main()
    