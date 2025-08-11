import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# === Training Function ===
def train_model(model, criterion, optimizer, train_loader, val_loader, device, epochs, history):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"🟢 Epoch [{epoch+1}/{epochs}]")
        for images, labels in loop:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # === Validation ===
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()

        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        print(f"✅ Validation Accuracy after Epoch {epoch+1}: {val_acc:.2f}%")

        # === ADDED: Save metrics for graph plotting later ===
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

# === Confusion Matrix ===
def plot_confusion_matrix(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation=90, cmap=plt.cm.Blues)
    plt.title("📊 Confusion Matrix (Validation Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix2.png")
    plt.close()

    # === ADDED: Final Score Bar Chart ===
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    precisions = [report[label]["precision"] for label in class_names]
    recalls = [report[label]["recall"] for label in class_names]
    f1_scores = [report[label]["f1-score"] for label in class_names]

    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1_scores, width, label="F1-score")
    ax.set_ylabel("Score")
    ax.set_xlabel("Class")
    ax.set_title("Final Classification Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig("final_scores.png")
    plt.close()

# === ADDED: Label distribution plot ===
def plot_label_distribution(train_dataset, val_dataset):
    train_counts = np.bincount([label for _, label in train_dataset.samples])
    val_counts = np.bincount([label for _, label in val_dataset.samples])
    class_names = train_dataset.classes

    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, train_counts, width, label="Train")
    ax.bar(x + width/2, val_counts, width, label="Validation")
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Class")
    ax.set_title("Label Distribution in Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig("label_distribution.png")
    plt.close()

# === ADDED: Training history plots ===
def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")
    plt.close()

# === Main logic ===
if __name__ == "__main__":
    train_dir = "dataset/train"
    val_dir = "dataset/valid"
    num_classes = 29
    batch_size = 32
    initial_epochs = 5
    fine_tune_epochs = 10
    learning_rate = 1e-4
    fine_tune_lr = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔧 Using device:", device)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    targets = [label for _, label in train_dataset.samples]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = models.mobilenet_v2(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = Adam(model.classifier.parameters(), lr=learning_rate)

    # === ADDED: History dict ===
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("🔄 Phase 1: Training only the classifier head...")
    train_model(model, criterion, optimizer, train_loader, val_loader, device, initial_epochs, history)

    print("🎯 Phase 2: Fine-tuning entire model...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = Adam(model.parameters(), lr=fine_tune_lr)
    train_model(model, criterion, optimizer, train_loader, val_loader, device, fine_tune_epochs, history)

    torch.save(model.state_dict(), "mobilenet70breeds_thingwei.pth")
    print("💾 Model saved as mobilenet70breeds_thingwei.pth")

    class_names = train_dataset.classes
    plot_confusion_matrix(model, val_loader, device, class_names)
    plot_label_distribution(train_dataset, val_dataset)
    plot_training_history(history)
