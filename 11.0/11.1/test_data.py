# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：test_data.py
# 开发时间：2025/12/20 02:49
# _*_ coding: UTF-8 _*_

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

DATA_DIR = "../data"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip().split()

def main():
    print("Loading datasets...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    print("\n===== Basic Info =====")
    print("Train size:", len(df_train))
    print("Test size :", len(df_test))

    print("\n===== Columns =====")
    print(df_train.columns.tolist())

    # Label distribution
    labels = df_train["label"].tolist()
    counter = Counter(labels)

    print("\n===== Label Distribution =====")
    for k in sorted(counter.keys()):
        print(f"Label {k}: {counter[k]}")

    # Text length analysis
    lengths = []
    for t in df_train["content"].fillna(""):
        lengths.append(len(simple_tokenize(t)))

    lengths = np.array(lengths)

    print("\n===== Text Length Statistics =====")
    print("Min length:", lengths.min())
    print("Mean length:", lengths.mean())
    print("90% quantile:", np.percentile(lengths, 90))
    print("95% quantile:", np.percentile(lengths, 95))
    print("Max length:", lengths.max())

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=100)
    plt.title("Text Length Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
