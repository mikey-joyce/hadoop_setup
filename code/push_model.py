from transformers import AutoModelForSequenceClassification
from ray.train import Checkpoint
from huggingface_hub import HfApi
    
def push_model_and_tokenizer():
    checkpoint_path = "/home/ubuntu/ray_results/twitter-roberta-finetune/TorchTrainer_4639e_00000_0_2025-05-02_15-36-54/checkpoint_000004/checkpoint"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    
    print("Model loaded successfully.")
    
    model.push_to_hub(
            repo_id="twitter_roberta_finetuned",
            commit_message="Trained for 4 epochs"
            )
    
    print("Model pushed to Hugging Face Hub successfully.")
    
    # push tokenizer

def push_checkpoint_directory(checkpoint_dir, repo_id, repo_type):
    api = HfApi()
    api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_id,
            repo_type=repo_type
            )

if __name__ == "__main__":
    # push_model_and_tokenizer()

    # push checkpoint dir
    checkpoint_dir = "~/ray_results/twitter-roberta-finetune/TorchTrainer_4639e_00000_0_2025-05-02_15-36-54/checkpoint_000004/checkpoint/"
    repo_id = "zayanhugsAI/twitter_roberta_finetuned"
    repo_type = "model"

    try:
        push_checkpoint_directory(checkpoint_dir, repo_id, repo_type)
    except Exception as e:
        print("Couldn't push checkpoint directory. Exception: ", e)

    print(f"Pushed checkpoint directory to {repo_id}!")


    
