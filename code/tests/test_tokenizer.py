from transformers import AutoTokenizer

def test_tokenizer(model: str, text: str):
    """
    Tokenize the provided text using the tokenizer from the given model.
    
    Args:
        model (str): Pre-trained model name or path.
        text (str): Input string to tokenize.
    """
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    
    tokenized_output = tokenizer(
        text,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )
    
    print("Tokenized Output:")
    print(tokenized_output)

if __name__ == "__main__":
    # Hardcoded model and text
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    sample_text = "Hello, world! This is a test string."
    test_tokenizer(model_name, sample_text)