#_*_coding:UTF-8_*_
#开发人员：NBT
#文件名称：submission.py
#开发时间：2025/12/22 13:29
# Generate submission.csv from trained best_model.pt
# -*- coding: utf-8 -*-
# submission.py
# Load best transformer model and predict test.csv -> submission.csv

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

from transformers import AutoTokenizer
from model_small import SentimentModel


# ================= Dataset =================
class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }


# ================= Main =================
def main():
    # -------- paths --------
    test_csv = "../data/test.csv"
    sample_csv = "../data/sample_submission.csv"
    model_path = "./saved_models/best_model.pt"
    out_csv = "./out/submission.csv"

    # -------- config --------
    model_name = "roberta-large"   # 必须与 train.py 一致
    num_labels = 5
    max_len = 80                       # 与训练一致
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- checks --------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found: {model_path}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError("test.csv not found")

    os.makedirs("./out", exist_ok=True)

    # -------- load data --------
    df_test = pd.read_csv(test_csv)
    test_texts = df_test["content"].fillna("").astype(str).tolist()

    # -------- tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_ds = TestDataset(test_texts, tokenizer, max_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    # -------- model --------
    model = SentimentModel(model_name, num_labels=num_labels)
    state = torch.load(model_path, map_location=device)

    # 兼容 DataParallel 保存的权重
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    model.load_state_dict(new_state)
    model.to(device)
    model.eval()

    # -------- inference --------
    preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            batch_preds = torch.argmax(logits, dim=-1)
            preds.extend(batch_preds.cpu().tolist())

    # -------- save submission --------
    submission = pd.DataFrame({
        "id": df_test["id"],
        "label": preds
    })
    submission.to_csv(out_csv, index=False)
    print(f"Submission saved to {out_csv}")

    # -------- pseudo accuracy --------
    if os.path.exists(sample_csv):
        df_sample = pd.read_csv(sample_csv)

        if "label" in df_sample.columns:
            df_pred = submission.sort_values("id").reset_index(drop=True)
            df_true = df_sample.sort_values("id").reset_index(drop=True)

            acc = accuracy_score(df_true["label"], df_pred["label"])
            print("=" * 30)
            print(f"Pseudo Test ACC = {acc:.4f}")
            print("=" * 30)
        else:
            print("sample_submission.csv has no label column, skip accuracy.")
    else:
        print("sample_submission.csv not found, skip accuracy.")


if __name__ == "__main__":
    main()
