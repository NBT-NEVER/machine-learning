# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：train.py
# 开发时间：2025/12/20 02:55
# train.py
# 5-Class Sentiment Analysis with RoBERTa-large

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from torch.optim import AdamW
# from model_small import SentimentModel
from model import SentimentModel

# =====================
# Seed
# =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =====================
# Dataset
# =====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# =====================
# Train / Eval
# =====================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds, trues = [], [], []
    for batch in tqdm(loader, desc="Val", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        preds.extend(torch.argmax(logits, -1).cpu().numpy())
        trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    return np.mean(losses), acc

# =====================
# Main
# =====================
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "roberta-large"
    max_len = 80
    batch_size = 16
    epochs = 3
    lr = 2e-5

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("out", exist_ok=True)

    df = pd.read_csv("../data/train.csv")
    texts = df["content"].fillna("").astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = TextDataset(tr_texts, tr_labels, tokenizer, max_len)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SentimentModel(model_name, num_labels=5).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "saved_models/best_model.pt")
            print("→ Best model saved")

        pd.DataFrame(history).to_csv("saved_models/training_log.csv", index=False)

    # Loss curve
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.legend()
    plt.savefig("out/loss_curve.png")

if __name__ == "__main__":
    main()
