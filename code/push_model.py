from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ray.train import Checkpoint
from huggingface_hub import HfApi
    
def push_model_and_tokenizer(checkpoint_path, repo_id, commit_message=None):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    
    print("Model loaded successfully.")
    
    model.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message if commit_message is not None else "trained")
    
    print("Model pushed to Hugging Face Hub successfully.")
    
    # push tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path) 
    tokenizer.push_to_hub(repo_id)

    print("Tokenizer pushed to huggingface hub successfully")

def push_checkpoint_directory(checkpoint_dir, repo_id, repo_type):
    api = HfApi()
    api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_id,
            repo_type=repo_type
            )

if __name__ == "__main__":
    # push model and tokenizer
    path = "/home/ubuntu/ray_results/zayanhugsAI/twitter_roberta_finetuned/TorchTrainer_f0b32_00000_0_2025-05-06_02-00-52/checkpoint_000002/checkpoint"

    # push checkpoint dir
    # checkpoint_dir = "~/ray_results/twitter-roberta-finetune/TorchTrainer_4639e_00000_0_2025-05-02_15-36-54/checkpoint_000004/checkpoint/"
    repo_id = "zayanhugsAI/twitter_roberta_finetuned_2"
    repo_type = "model"
    commit_message = "Trained for another 3 epochs"

    # push_checkpoint_directory(checkpoint_dir, repo_id, repo_type)
    push_model_and_tokenizer(path, repo_id)
    

    print(f"Pushed checkpoint directory to {repo_id}!")


    
