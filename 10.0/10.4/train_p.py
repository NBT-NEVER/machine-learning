# -*- coding: utf-8 -*-
# 开发人员：NBT
# 文件名称：train_p_fixed.py
# 开发时间：2025/12/11

"""
可稳定收敛版本（干净、可靠、和 model2 完全兼容）
仅包含最必要的增强：
- Embedding 可微调
- 正确的 vocab + padding
- 固定 max_len 截断
- GloVe 加载无误
- 正常训练 / 验证循环
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model2 import MySentimentModel, load_glove_embeddings


# ==============================================================
# Utils
# ==============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simple_tokenize(text):
    """英文任务，不考虑中文"""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip().split()


# ==============================================================
# Dataset
# ==============================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_index, max_len):
        self.labels = labels
        self.word_index = word_index
        self.max_len = max_len
        self.sequences = [self.text_to_seq(t) for t in texts]

    def text_to_seq(self, text):
        tokens = simple_tokenize(text)
        seq = [self.word_index.get(tok, 1) for tok in tokens]  # 1 = <unk>, 0 = pad
        if len(seq) == 0:
            seq = [1]

        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq += [0] * (self.max_len - len(seq))

        return seq

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.labels is None:
            return x
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ==============================================================
# Training / Eval
# ==============================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    trues = []

    for x, y in tqdm(loader, desc="Val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds.extend(torch.argmax(logits, -1).cpu().numpy())
        trues.extend(y.cpu().numpy())

    return float(np.mean(losses)), accuracy_score(trues, preds)


# ==============================================================
# Main training function
# ==============================================================
def run_training(
    train_path="../data/train.csv",
    test_path="../data/test.csv",
    sample_path="../data/sample_submission.csv",
    out_path="./submission.csv",
    embed_dim=300,
    hidden_dim=256,
    L2=1e-5,
    batch_size=32,
    epochs=3,
    max_len=900,
    lr=8e-4,
    glove_path="../data/glove.6B.300d.txt",
    seed=42,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    set_seed(seed)

    # -------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_sample = pd.read_csv(sample_path)

    texts = df_train["content"].fillna("").astype(str).tolist()
    labels = df_train["label"].astype(int).tolist()

    # -------------------------------------------------------------
    # Build Vocabulary
    # -------------------------------------------------------------
    print("Building vocab...")
    all_tokens = []
    for t in texts:
        all_tokens.extend(simple_tokenize(t))

    vocab = sorted(list(set(all_tokens)))
    word_index = {w: i + 2 for i, w in enumerate(vocab)}  # 0 = pad, 1 = unk
    word_index["<unk>"] = 1
    vocab_size = len(word_index) + 1

    os.makedirs("./saved_models", exist_ok=True)
    torch.save(word_index, "./saved_models/vocab.pt")

    print(f"Vocab size = {vocab_size}")

    # -------------------------------------------------------------
    # Load GloVe
    # -------------------------------------------------------------
    print("Loading GloVe...")
    embedding_matrix = load_glove_embeddings(glove_path, word_index, embed_dim)
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

    # -------------------------------------------------------------
    # Split Dataset
    # -------------------------------------------------------------
    tr_x, val_x, tr_y, val_y = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=seed
    )

    train_ds = TextDataset(tr_x, tr_y, word_index, max_len)
    val_ds = TextDataset(val_x, val_y, word_index, max_len)
    test_ds = TextDataset(df_test["content"].fillna("").astype(str).tolist(),
                          None, word_index, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # -------------------------------------------------------------
    # Model / Optimizer
    # -------------------------------------------------------------
    model = MySentimentModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        embedding_matrix=embedding_matrix
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=L2
    )
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    best_acc = 0.0
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        print(f"\n======== Epoch {epoch}/{epochs} ========")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./saved_models/best.pt")
            print("Saved best model.")

    # -------------------------------------------------------------
    # Load best model for prediction
    # -------------------------------------------------------------
    model.load_state_dict(torch.load("./saved_models/best.pt"))
    model.eval()

    preds = []
    with torch.no_grad():
        for x in tqdm(test_loader, desc="Predict"):
            x = x.to(device)
            logits = model(x)
            p = torch.argmax(logits, -1).cpu().numpy().tolist()
            preds.extend(p)

    df_pred = pd.DataFrame({"id": df_test["id"], "label": preds})
    df_pred.to_csv(out_path, index=False)
    print("Saved test predictions:", out_path)

    return history


# ==============================================================
# CLI
# ==============================================================
if __name__ == "__main__":
    run_training()
