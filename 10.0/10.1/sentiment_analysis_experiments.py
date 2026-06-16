"""
Sentiment Analysis Experiments — BERT (Best Practice Version)
Compatible with: transformers 4.57.3, PyTorch (GPU/CPU), fp16

Features:
- Train/Val split, training logs (loss & lr)
- Save best model
- Draw loss curve (loss_curve.png)
- Evaluate test accuracy using sample_submission.csv (ground truth)
- Save submission.csv

"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ========== Utility ==========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": str(self.texts[idx])}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


def collate_fn(batch, tokenizer, max_len):
    texts = [x["text"] for x in batch]
    enc = tokenizer(texts, padding="longest", truncation=True,
                    max_length=max_len, return_tensors="pt")

    if "label" in batch[0]:
        enc["labels"] = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    return enc


# ========== Training ==========
def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
    model.train()
    losses = []
    lrs = []

    for batch in tqdm(dataloader, desc="Train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        losses.append(loss.item())
        lrs.append(scheduler.get_last_lr()[0])

    return np.mean(losses), np.mean(lrs)


@torch.no_grad()
def eval_model(model, dataloader, device):
    model.eval()
    losses = []
    preds = []
    trues = []

    for batch in tqdm(dataloader, desc="Val", leave=False):
        labels = batch["labels"].to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, -1).cpu().numpy())
        trues.extend(labels.cpu().numpy())

    return np.mean(losses), accuracy_score(trues, preds)


# ========== Test prediction ==========
@torch.no_grad()
def predict_test(model, dataloader, device):
    model.eval()
    preds = []

    for batch in tqdm(dataloader, desc="Predict"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds.extend((probs >= 0.5).int().cpu().numpy())

    return preds


# ========== Main ==========
def run_bert(train_path, test_path, out_path, sample_path,
             model_name="bert-base-uncased", epochs=3, batch_size=16,
             lr=2e-5, max_len=128, weight_decay=0.01, seed=42,
             device=None, fp16=True):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    set_seed(seed)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_sample = pd.read_csv(sample_path)   # real labels

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Split
    texts = df_train["content"].fillna("").tolist()
    labels = df_train["label"].tolist()

    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=seed
    )

    # Dataset & Loader
    train_ds = TextDataset(tr_texts, tr_labels)
    val_ds = TextDataset(val_texts, val_labels)
    test_ds = TextDataset(df_test["content"].fillna("").tolist())

    collate = lambda b: collate_fn(b, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.06 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device == "cuda"))

    # ========== Training loop ==========
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")

        train_loss, lr_now = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_loss, val_acc = eval_model(model, val_loader, device)

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)

        # Save best model (in memory)
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        #     print("→ New best model saved")

        # Save best model (to local file)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            save_path = f"./saved_models/{model_name}_best.pt"
            os.makedirs("./saved_models", exist_ok=True)

            torch.save(model.state_dict(), save_path)
            print(f"→ New best model saved to {save_path}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # ========== Save training log ==========
    pd.DataFrame(history).to_csv("training_log.csv", index=False)
    print("Training log saved to training_log.csv")

    # ========== Draw loss curve ==========
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("Loss curve saved as loss_curve.png")

    # ========== Predict test ==========
    preds = predict_test(model, test_loader, device)

    submission = pd.DataFrame({"id": df_test["id"], "label": preds})
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")

    # ========== Test Accuracy ==========
    df_pred = submission.sort_values("id")
    df_true = df_sample.sort_values("id")

    test_acc = accuracy_score(df_true["label"], df_pred["label"])
    print(f"\n==============================")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"==============================")

    return test_acc


# ========== CLI ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--sample", required=True, help="sample_submission.csv containing real labels")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_fp16", action="store_true")

    if len(sys.argv) == 1:
        class DefaultArgs:
            train = "../data/train.csv"
            test = "../data/test.csv"
            sample = "../data/sample_submission.csv"
            out = "submission.csv"
            model_name = "roberta-base"
            epochs = 2
            batch_size = 16
            lr = 2e-5
            max_len = 512
            weight_decay = 0.02
            seed = 42
            no_fp16 = False

        args = DefaultArgs()
    else:
        args = parser.parse_args()

    run_bert(
        args.train, args.test, args.out, args.sample,
        model_name=args.model_name,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, max_len=args.max_len,
        weight_decay=args.weight_decay,
        seed=args.seed,
        fp16=(not args.no_fp16)
    )
