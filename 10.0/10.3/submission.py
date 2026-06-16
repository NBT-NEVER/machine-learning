# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：model2.py
# 开发时间：2025/12/20 2:38
# -*- coding: UTF-8 -*-
# model2.py
# submission.py
# Load saved vocab & saved model (saved_models/1.pt) then predict test.csv -> submission.csv

import os
import torch
import pandas as pd
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

from model import MySentimentModel  # same model definition used in train.py

def simple_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip().split()

class InferDataset(Dataset):
    def __init__(self, texts, word_index, max_len):
        self.word_index = word_index
        self.max_len = max_len
        self.sequences = [self.text_to_sequence(t) for t in texts]

    def text_to_sequence(self, text):
        tokens = simple_tokenize(text)
        seq = [self.word_index.get(tok, 0) for tok in tokens]
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
        return torch.tensor(self.sequences[idx], dtype=torch.long)

def collate_fn(batch):
    # batch is list of tensors
    xs = torch.stack(batch, dim=0)
    return xs

def main():
    # defaults
    test_csv = "../data/test.csv"
    sample_csv = "../data/sample_submission.csv"
    out_csv = "./out/s_submission.csv"
    vocab_path = "./saved_models/vocab.pt"
    model_path = "./saved_models/1.pt"
    max_len = 900
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # checks
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab not found: {vocab_path}. Run train.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}. Run train.py first.")

    # load resources
    word_index = torch.load(vocab_path)
    df_test = pd.read_csv(test_csv)
    test_texts = df_test["content"].fillna("").astype(str).tolist()

    # load model
    vocab_size = len(word_index) + 1
    # hyperparams: must match training defaults (embed_dim, hidden_dim)
    embed_dim = 300
    hidden_dim = 256
    model = MySentimentModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=2, embedding_matrix=None)
    state = torch.load(model_path, map_location=device)
    # adapt possible "module." prefix
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state)
    model.to(device)
    model.eval()

    # prepare dataloader
    infer_ds = InferDataset(test_texts, word_index, max_len)
    loader = DataLoader(infer_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    preds = []
    with torch.no_grad():
        for x in tqdm(loader, desc="Predict"):
            x = x.to(device)
            logits = model(x)
            p = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend(p)

    submission = pd.DataFrame({"id": df_test["id"], "label": preds})
    submission.to_csv(out_csv, index=False)
    print(f"Saved submission to {out_csv}")

    # if sample present, compute accuracy
    if os.path.exists(sample_csv):
        df_sample = pd.read_csv(sample_csv)
        if "label" in df_sample.columns:
            df_pred = submission.sort_values("id").reset_index(drop=True)
            df_true = df_sample.sort_values("id").reset_index(drop=True)
            acc = accuracy_score(df_true["label"], df_pred["label"])
            print(f"Test Accuracy: {acc:.4f}")
        else:
            print("sample_submission.csv exists but has no 'label' column; skipping accuracy.")
    else:
        print("sample_submission.csv not found; skipping accuracy.")

if __name__ == "__main__":
    main()




