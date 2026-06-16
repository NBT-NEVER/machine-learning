# -*- coding: utf-8 -*-
# 开发人员：NBT
# 文件名称：train.py
# 开发时间：2025/12/09

# -*- coding: utf-8 -*-
# 开发人员：NBT
# 文件名称：train.py
# 开发时间：2025/12/09

"""
Train Sentiment Classifier (NO Pretrained Models)
Uses GloVe + LSTM model defined in model.py

Features added:
- print & save training parameters (params.json)
- save training_log.csv (per epoch train_loss/val_loss/val_acc)
- draw loss curve (loss_curve.png)
- save vocab to saved_models/vocab.pt
- save best model to saved_models/1.pt
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import MySentimentModel, load_glove_embeddings

# ============================================
# Utils
# ============================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def simple_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip().split()

# ============================================
# Dataset
# ============================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_index, max_len):
        self.labels = labels
        self.max_len = max_len
        self.word_index = word_index
        self.sequences = [self.text_to_sequence(x) for x in texts]

    def text_to_sequence(self, text):
        tokens = simple_tokenize(text)
        seq = []
        for tok in tokens:
            if tok in self.word_index:
                seq.append(self.word_index[tok])
        if len(seq) == 0:
            seq = [0]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = seq + [0] * (self.max_len - len(seq))
        return seq

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x

# ============================================
# Training helpers
# ============================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y in tqdm(dataloader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    losses = []
    preds = []
    trues = []
    for x, y in tqdm(dataloader, desc="Val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds.extend(torch.argmax(logits, -1).cpu().numpy().tolist())
        trues.extend(y.cpu().numpy().tolist())
    val_loss = float(np.mean(losses)) if losses else 0.0
    val_acc = accuracy_score(trues, preds) if len(trues) else 0.0
    return val_loss, val_acc

# ============================================
# Main training function
# ============================================
def run_training(train_path="train.csv",
                 test_path="test.csv",
                 sample_path="sample_submission.csv",
                 out_path="submission.csv",
                 L2 = 1e-3,
                 embed_dim=300,
                 hidden_dim=256,
                 batch_size=32,
                 epochs=5,
                 max_len=200,
                 lr=1e-3,
                 glove_path="glove.6B.300d.txt",
                 seed=42,
                 device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    set_seed(seed)

    # --- load data ---
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_sample = pd.read_csv(sample_path)

    texts = df_train["content"].fillna("").astype(str).tolist()
    labels = df_train["label"].astype(int).tolist()

    # --- build vocab ---
    print("Building vocabulary...")
    all_tokens = []
    for t in texts:
        all_tokens.extend(simple_tokenize(t))
    vocab_list = sorted(list(set(all_tokens)))
    word_index = {w: i+1 for i, w in enumerate(vocab_list)}  # 0 reserved for PAD
    vocab_size = len(word_index) + 1
    print(f"Vocab size: {vocab_size}")

    # --- save params and hyperparameters ---
    params = {
        "train_path": train_path,
        "test_path": test_path,
        "sample_path": sample_path,
        "out_path": out_path,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "L2": L2,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_len": max_len,
        "lr": lr,
        "glove_path": glove_path,
        "seed": seed,
        "vocab_size": vocab_size
    }
    os.makedirs("./saved_models", exist_ok=True)
    with open("./saved_models/params.json", "w", encoding="utf-8") as fp:
        json.dump(params, fp, indent=2)
    print("Saved training parameters to ./saved_models/params.json")
    # also print them
    print(json.dumps(params, indent=2))

    # --- load glove embeddings (initialize embedding matrix) ---
    if not os.path.exists(glove_path):
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")
    embedding_matrix = load_glove_embeddings(glove_path, word_index, embed_dim)  # returns torch tensor

    # --- train/val split ---
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=seed
    )

    # --- datasets & loaders ---
    train_ds = TextDataset(tr_texts, tr_labels, word_index, max_len)
    val_ds = TextDataset(val_texts, val_labels, word_index, max_len)
    test_ds = TextDataset(df_test["content"].fillna("").astype(str).tolist(), None, word_index, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- model, optimizer, criterion ---
    model = MySentimentModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        embedding_matrix=embedding_matrix
    ).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=L2  # ← L2 正则化
    )

    criterion = nn.CrossEntropyLoss()

    # --- save vocab for inference ---
    torch.save(word_index, "./saved_models/vocab.pt")
    print("Saved vocab to ./saved_models/vocab.pt")

    best_val_acc = 0.0
    save_path = "./saved_models/1.pt"  # final saved best model
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    # --- training loop ---
    for epoch in range(1, epochs+1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"→ Best model saved to {save_path}")

        # save per-epoch training log
        pd.DataFrame(history).to_csv("./saved_models/training_log.csv", index=False)

    # --- restore best model ---
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Loaded best model for final prediction.")
    else:
        print("No saved model found; using current model for prediction.")

    # --- draw loss curve ---
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./saved_models/loss_curve.png")
    print("Saved loss curve to ./saved_models/loss_curve.png")

    # --- predict test ---
    preds = []
    model.eval()
    with torch.no_grad():
        for x in tqdm(test_loader, desc="Predict"):
            x = x.to(device)
            logits = model(x)
            p = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend(p)

    submission = pd.DataFrame({"id": df_test["id"], "label": preds})
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")

    # --- evaluate test accuracy if sample available ---
    df_pred = submission.sort_values("id").reset_index(drop=True)
    df_true = df_sample.sort_values("id").reset_index(drop=True)
    if "label" in df_true.columns:
        test_acc = accuracy_score(df_true["label"], df_pred["label"])
        print("\n==============================")
        print(f"Test ACC = {test_acc:.4f}")
        print("==============================")
    else:
        print("sample_submission does not contain 'label' column; skipping test accuracy.")

    # also save history file
    pd.DataFrame(history).to_csv("./saved_models/training_log.csv", index=False)
    print("Training log saved to ./saved_models/training_log.csv")

    return history

# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    # default args (you can edit here)
    class Args:
        train = "../data/train.csv"
        test = "../data/test.csv"
        sample = "../data/sample_submission.csv"
        out = "./out/submission.csv"
        L2 = 1e-5
        epochs = 3
        batch_size = 32
        max_len = 900
        seed = 42
        embed_dim = 300
        hidden_dim = 256
        lr = 4.65e-4
        glove_path = "../data/glove.6B.300d.txt"

    args = Args()
    run_training(
        train_path=args.train,
        test_path=args.test,
        sample_path=args.sample,
        out_path=args.out,
        L2=args.L2,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_len=args.max_len,
        lr=args.lr,
        glove_path=args.glove_path,
        seed=args.seed
    )
