from transformers import Trainer, TrainingArguments

model = ViTWithLabelReps(
    model_name="google/vit-base-patch16-224-in21k",
    num_classes=10,  # e.g., CIFAR-10
    embed_dim=256,
    use_normalization=True,
    reg_strength=1e-3,
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=feature_extractor,  # HuggingFace ViT uses an image processor
    compute_metrics=compute_metrics,
)

trainer.train()