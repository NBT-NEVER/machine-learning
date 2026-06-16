# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：model_small.py
# 开发时间：2025/12/20 03:03


# -*- coding: utf-8 -*-
# model_small.py
# Fast pretrained model for multi-class sentiment analysis

import torch
import torch.nn as nn
from transformers import AutoModel

class SentimentModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 5):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # DistilRoBERTa / BERT / RoBERTa 通用写法
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)

        logits = self.classifier(cls_rep)
        return logits
