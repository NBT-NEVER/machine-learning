# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：test_glove.py
# 开发时间：2025/12/20 02:50
# _*_ coding: UTF-8 _*_

import pandas as pd
import re

DATA_DIR = "../data"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
GLOVE_PATH = f"{DATA_DIR}/glove.6B.300d.txt"

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip().split()

def load_glove_vocab(glove_path):
    glove_vocab = set()
    print("Loading GloVe vocabulary...")
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.split(" ")[0]
            glove_vocab.add(word)
    return glove_vocab

def main():
    print("Loading training data...")
    df = pd.read_csv(TRAIN_PATH)

    print("Building vocabulary from training set...")
    vocab = set()
    for t in df["content"].fillna(""):
        vocab.update(simple_tokenize(t))

    print("Vocabulary size:", len(vocab))

    glove_vocab = load_glove_vocab(GLOVE_PATH)

    hit = 0
    for w in vocab:
        if w in glove_vocab:
            hit += 1

    coverage = hit / len(vocab)

    print("\n===== GloVe Coverage Analysis =====")
    print(f"Vocabulary size      : {len(vocab)}")
    print(f"GloVe matched words  : {hit}")
    print(f"Coverage rate        : {coverage:.4f}")

if __name__ == "__main__":
    main()
