# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：glove_test.py
# 开发时间：2025/12/11 22:35
# coding: utf-8
# coding: utf-8
import pandas as pd
import numpy as np
import re

############################################################
# 1. 基础配置
############################################################
train_path = "../data/train.csv"
glove_path = "../data/glove.6B.300d.txt"
embed_dim = 300

############################################################
# 2. 简单英文 tokenizer
############################################################
def simple_tokenize(text):
    text = str(text).lower()
    tokens = re.findall(r"[a-z]+", text)
    return tokens

############################################################
# 3. 加载训练数据
############################################################
df_train = pd.read_csv(train_path)

texts = df_train["content"].astype(str).tolist()
print("Train size:", len(texts))
print("Example text:", texts[0][:200])

############################################################
# 4. 构建词表
############################################################
from collections import Counter
counter = Counter()
for t in texts:
    counter.update(simple_tokenize(t))

vocab = ["<pad>", "<unk>"] + [w for w, c in counter.items() if c >= 2]
word2id = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print("Vocab size:", vocab_size)

############################################################
# 5. 加载 GloVe 词向量
############################################################
embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)

print("Loading GloVe...")
with open(glove_path, "r", encoding="utf8") as f:
    for line in f:
        parts = line.rstrip().split()
        word = parts[0]
        if word in word2id:
            embedding_matrix[word2id[word]] = np.asarray(parts[1:], dtype=float)

print("Done loading GloVe.")

############################################################
# 6. 打印词向量命中率
############################################################
found = np.sum(np.any(embedding_matrix != 0, axis=1))
print("Embedding hit rate:", found, "/", vocab_size, "=", found / vocab_size)

############################################################
# 7. 第一个样本 tokens
############################################################
print("\nTokens of the first text:")
print(simple_tokenize(texts[0]))

############################################################
# 8. 句子长度统计
############################################################
lengths = [len(simple_tokenize(t)) for t in texts]
print("\nAvg length:", np.mean(lengths))
print("Max length:", np.max(lengths))

############################################################
# 9. 标签分布
############################################################
print("\nLabel distribution:")
print(df_train["label"].value_counts())
