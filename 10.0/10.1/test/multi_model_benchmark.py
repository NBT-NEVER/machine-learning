# -*- coding: utf-8 -*-
import os
import csv
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


###########################################
#          模型列表（按精度排序）
###########################################
model_list= [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    # "albert-base-v2",
]

###########################################
#              数据集定义
###########################################
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.texts)


###########################################
#               训练函数
###########################################
def train_one(model_name, train_loader, val_loader, device, save_dir):
    print(f"\n\n==============================")
    print(f"▶ Training model: {model_name}")
    print(f"==============================\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * 3

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    model_save_path = os.path.join(save_dir, f"{model_name.replace('/', '_')}_best.pt")

    for epoch in range(1, 4):
        print(f"\n===== Epoch {epoch}/3 =====")
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc="Train"):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                output = model(**batch)
                loss = output.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ----------- 验证 ----------------
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda"):
                    output = model(**batch)

                loss = output.loss
                val_loss += loss.item()

                preds = torch.argmax(output.logits, dim=1)
                labels = batch["labels"]
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print("→ New best model saved")

    # ---------- 绘图 ----------
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.legend()
    plt.title(model_name + " Loss Curve")
    plt.savefig(os.path.join(save_dir, f"{model_name.replace('/', '_')}_loss.png"))
    plt.clf()

    plt.plot(history["val_acc"], label="val accuracy")
    plt.legend()
    plt.title(model_name + " Val Acc")
    plt.savefig(os.path.join(save_dir, f"{model_name.replace('/', '_')}_acc.png"))
    plt.clf()

    # ---------- 保存日志 ----------
    log_csv = os.path.join(save_dir, f"{model_name.replace('/', '_')}_log.csv")
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        for i in range(3):
            writer.writerow([
                i+1,
                history["train_loss"][i],
                history["val_loss"][i],
                history["val_acc"][i]
            ])

    return best_acc


###########################################
#             多模型主函数
###########################################
def run_multi_model_benchmark(train_texts, train_labels, val_texts, val_labels):

    save_dir = "benchmark_results"
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("\n\n======= START MULTI-MODEL TRAINING =======\n")

    results = {}

    for model_name in model_list:

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_ds = MyDataset(train_texts, train_labels, tokenizer)
        val_ds = MyDataset(val_texts, val_labels, tokenizer)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        acc = train_one(model_name, train_loader, val_loader, device, save_dir)
        results[model_name] = acc

    print("\n\n======= RESULTS =======")
    for m, a in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{m:35s}  {a:.4f}")
