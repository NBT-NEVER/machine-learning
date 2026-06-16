# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：submission.py
# 开发时间：2025/12/23 18:32
# _*_coding:UTF-8_*_
# submission.py
# Binary sentiment prediction with DeBERTa-v3-base

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

from train import TRAIN_CONFIG
from model import SentimentModel


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
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def main():
    cfg = TRAIN_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_csv   = "../data/test.csv"
    sample_csv = "../data/sample_submission.csv"
    out_csv    = "./out/submission.csv"

    model_path = os.path.join(
        cfg["save_dir"],
        f"best_model_{cfg['model_name'].replace('/', '-')}_binary.pt"
    )

    os.makedirs("./out", exist_ok=True)

    df_test = pd.read_csv(test_csv)
    texts = df_test["content"].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    test_ds = TestDataset(texts, tokenizer, cfg["max_len"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = SentimentModel(cfg["model_name"], cfg["num_labels"])
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predict"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds.extend(torch.argmax(logits, -1).cpu().numpy())

    submission = pd.DataFrame({"id": df_test["id"], "label": preds})
    submission.to_csv(out_csv, index=False)
    print(f"Submission saved to {out_csv}")

    if os.path.exists(sample_csv):
        df_sample = pd.read_csv(sample_csv)
        acc = accuracy_score(df_sample["label"], submission["label"])
        print(f"Pseudo Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
