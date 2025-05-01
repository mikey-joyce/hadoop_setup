def train_func(config: dict):
    """
    Main training function to be executed by Ray.
    This function largely follows the train_func from https://docs.ray.io/en/latest/train/examples/transformers/huggingface_text_classification.html#hf-train,
    which takes from https://huggingface.co/docs/transformers/en/training#trainer
    """
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    # Get datasets from config
    train_dataset = config.get("train")
    val_dataset = config.get("val")
    
    if train_dataset is None:
        raise ValueError("Training dataset is None. Please provide a valid dataset.")
    if val_dataset is None:
        choice = input("No validation dataset provided. Restart or continue? (r/c): ")
        if choice.lower() == "r":
            raise ValueError("Validation dataset is None. Please provide a valid dataset.")
        elif choice.lower() == "c":
            print("Continuing without validation dataset.")
        val_dataset = None

    # Calculate maximum steps per epoch
    batch_size = config.get("batch_size", 16)
    learning_rate = config.get("learning_rate", 2e-5)
    epochs = config.get("epochs", 3)
    weight_decay = config.get("weight_decay", 0.01)
    
    # Compute maximum steps per epoch
    num_workers = config.get("num_workers", 1)
    max_steps_per_epoch = train_dataset.count() // (batch_size * num_workers)
    print(f"Max steps per epoch: {max_steps_per_epoch}")
    
    # Load tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Get label count
    num_labels, unique_labels = count_unique_labels(train_dataset)
    print(f"Detected {num_labels} unique labels: {unique_labels}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Create data collator using partial
    data_collator = partial(collate_fn, tokenizer=tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        logging_dir="./logs",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="f1" if val_dataset else None,
        fp16=True,  # Mixed precision training
        push_to_hub=False,
        disable_tqdm=True,  # Cleaner output in distributed environments
        report_to="none",
    )
    
    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics  # Add metrics computation
    )
    
    
    
    trainer.add_callback(RayTrainReportCallback()) # Use add_callback to report metrics to Ray
    trainer = prepare_trainer(trainer) # Use prepare_trainer to validate config
    
    
    # Train the model
    print("Starting training")
    trainer.train()
    
    # Save model and tokenizer
    model_path = "./sentiment_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Return results
    if val_dataset:
        eval_results = trainer.evaluate()
        return eval_results
    return {"status": "completed"}
