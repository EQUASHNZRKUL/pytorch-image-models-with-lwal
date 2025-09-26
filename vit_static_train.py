# from datasets import load_dataset
from transformers import AutoImageProcessor
from ViTWithLabelReps import ViTWithLabelReps, ViTWithStaticLabelReps

# HuggingFace ViT requires pixel pre-processing
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=processor.image_mean, std=processor.image_std),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

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

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTWithStaticLabelReps("google/vit-base-patch16-224-in21k", label_reps).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(5):  # example: 5 epochs
    model.train()
    total_loss, total_correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, loss = model(imgs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()

    train_loss = total_loss / len(train_loader.dataset)
    train_acc = total_correct / len(train_loader.dataset)

    # Eval
    model.eval()
    total_correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

    test_acc = total_correct / total
    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

# from transformers import TrainingArguments, Trainer

# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=64,
#     per_device_eval_batch_size=64,
#     num_train_epochs=10,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=50,
#     load_best_model_at_end=True,
# )

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = logits.argmax(axis=-1)
#     acc = (preds == labels).mean()
#     return {"accuracy": acc}

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_loader,
#     eval_dataset=test_loader,
#     data_collator=collate_fn,
#     tokenizer=processor,
#     compute_metrics=compute_metrics,
# )

# trainer.train()