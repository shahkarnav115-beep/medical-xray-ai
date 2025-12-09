import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from model import build_model
from collections import Counter

DATA_DIR = "data"

def train_model():

    # ---- TRANSFORMS ----
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # ---- DATASET ----
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

    # ---- MODEL ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)

    # Freeze feature extractor (recommended)
    for p in model.features.parameters():
        p.requires_grad = False

    # ---- CLASS WEIGHTS ----
    counts = Counter(train_ds.targets)
    normal = counts[0]
    pneu = counts[1]
    class_weights = torch.tensor([1/normal, 1/pneu], device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0

    # ---- TRAINING ----
    for epoch in range(40):
        model.train()
        total, correct = 0, 0

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # ---- VALIDATION ----
        model.eval()
        val_total, val_correct = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/40 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            torch.save(model.state_dict(), "models/best_model_vgg19.pt")
            best_acc = val_acc
            print("🔥 Saved new best model!")
