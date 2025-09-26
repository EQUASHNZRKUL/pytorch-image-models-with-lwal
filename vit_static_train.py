from datasets import load_dataset
from transformers import AutoImageProcessor
from ViTWithLabelReps import ViTWithLabelReps, ViTWithStaticLabelReps

# Load CIFAR-10
dataset = load_dataset("cifar10")

# HuggingFace ViT requires pixel pre-processing
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def transform(example):
    example["pixel_values"] = processor(images=example["img"], return_tensors="pt")["pixel_values"][0]
    return example

dataset = dataset.with_transform(transform)

# Define DataCollator
def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# Define label representation
import torch
num_classes = 10
label_reps = torch.eye(num_classes)  # [10, 10]

# Build the model
model = ViTWithStaticLabelReps(
    model_name="google/vit-base-patch16-224-in21k",
    label_reps=label_reps,
    use_normalization=True,
)

# Set up and run trainer
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collate_fn,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

trainer.train()