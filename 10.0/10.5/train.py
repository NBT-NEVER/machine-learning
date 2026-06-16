# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：train.py
# 开发时间：2025/12/23 18:32
# _*_coding:UTF-8_*_
# Train DeBERTa-v3-base for Binary Sentiment Classification

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import SentimentModel


# =====================
# Config
# =====================
TRAIN_CONFIG = {
    "model_name": "roberta-base",
    "num_labels": 2,
    "max_len": 512,
    "batch_size": 8,
    "epochs": 3,
    "lr": 2e-5,
    "seed": 42,
    "save_dir": "./saved_models"
}


# =====================
# Utils
# =====================
def set_seed(seed):
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
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =====================
# Train / Eval
# =====================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds, trues = [], [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, -1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())

    return np.mean(losses), accuracy_score(trues, preds)


# =====================
# Main
# =====================
def main():
    cfg = TRAIN_CONFIG
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_csv = "../data/train.csv"

    df = pd.read_csv(train_csv)
    texts = df["content"].fillna("").astype(str).tolist()
    labels = df["label"].tolist()

    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels,
        test_size=0.1,
        stratify=labels,
        random_state=cfg["seed"]
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    train_ds = TextDataset(tr_texts, tr_labels, tokenizer, cfg["max_len"])
    val_ds   = TextDataset(val_texts, val_labels, tokenizer, cfg["max_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = SentimentModel(cfg["model_name"], cfg["num_labels"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    os.makedirs(cfg["save_dir"], exist_ok=True)
    save_path = os.path.join(
        cfg["save_dir"],
        f"best_model_{cfg['model_name'].replace('/', '-')}_binary.pt"
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(cfg["epochs"]):
        print(f"\n===== Epoch {epoch+1}/{cfg['epochs']} =====")

        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(f"Train loss: {tr_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"→ Best model saved to {save_path}")

    pd.DataFrame(history).to_csv(
        os.path.join(cfg["save_dir"], "training_log_binary.csv"),
        index=False
    )

    plt.figure()
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["save_dir"], "loss_curve_binary.png"))

    print("Training finished.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
