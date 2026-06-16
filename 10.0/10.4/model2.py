# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：model2.py
# 开发时间：2025/12/11 18:33
# -*- coding: UTF-8 -*-
# model2.py
# 更强但不过拟合的版本：Embedding → 双层 BiLSTM → 轻量 Attention → Dropout → FC
# 开发人员：NBT（新版结构）

import torch
import torch.nn as nn
import numpy as np

# ============================================================
#           可调参数（与 model1 完全一致，保持兼容）
# ============================================================
DEFAULT_EMBED_DIM = 300
DEFAULT_HIDDEN_DIM = 256
DEFAULT_LSTM_LAYERS = 2
DEFAULT_EMBED_DROPOUT = 0.15      # 新增：对 embedding dropout，减少过拟合
DEFAULT_ATTEN_DROPOUT = 0.25      # attention dropout
DEFAULT_FC_DROPOUT = 0.40         # 全连接 dropout（最有效）
DEFAULT_LSTM_DROPOUT = 0.10       # LSTM 层间 dropout（适度，不会破坏序列）

# ============================================================
#   Load GloVe Embeddings（保持与你原有项目完全一致）
# ============================================================
def load_glove_embeddings(glove_path, word_index, embed_dim=300):
    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim), dtype=np.float32)

    print("Loading GloVe embeddings (this may take some time)...")
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split(" ")
            word = values[0]
            if word not in word_index:
                continue
            vector = np.asarray(values[1:], dtype=np.float32)
            embedding_matrix[word_index[word]] = vector

    print("GloVe loaded. Shape:", embedding_matrix.shape)
    return torch.tensor(embedding_matrix, dtype=torch.float32)


# ============================================================
#            轻量 Attention（避免过拟合）
# ============================================================
class LightAttention(nn.Module):
    """
    轻量 attention：只使用一层线性，参数量远小于原版 AttentionPooling
    不会像重 attention 那样严重过拟合
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(DEFAULT_ATTEN_DROPOUT)

    def forward(self, H):
        score = torch.tanh(self.attn(H))           # [B, L, 1]
        attn_weights = torch.softmax(score, dim=1)
        context = torch.sum(attn_weights * H, dim=1)
        return self.dropout(context)


# ============================================================
#         Model2: Embedding → BiLSTM → Attention → FC
# ============================================================
class MySentimentModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=DEFAULT_EMBED_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        num_classes=2,
        embedding_matrix=None
    ):
        super().__init__()

        # -----------------------------
        # 1. Embedding + dropout
        # -----------------------------
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            print("Initializing embedding with GloVe...")
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = True
        self.embed_dropout = nn.Dropout(DEFAULT_EMBED_DROPOUT)

        # -----------------------------
        # 2. 双层 BiLSTM（轻量 dropout）
        # -----------------------------
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=DEFAULT_LSTM_LAYERS,
            batch_first=True,
            dropout=DEFAULT_LSTM_DROPOUT,
            bidirectional=True
        )

        # -----------------------------
        # 3. 轻量 Attention（不会过拟合）
        # -----------------------------
        self.attention = LightAttention(hidden_dim)

        # -----------------------------
        # 4. FC 前 dropout（非常重要）
        # -----------------------------
        self.fc_dropout = nn.Dropout(DEFAULT_FC_DROPOUT)

        # -----------------------------
        # 5. 分类层
        # -----------------------------
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, mask=None):
        # x: [B, L]
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)

        H, _ = self.lstm(emb)           # [B, L, 2H]
        context = self.attention(H)     # [B, 2H]

        out = self.fc_dropout(context)
        logits = self.fc(out)

        return logits
