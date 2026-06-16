
# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：model.py
# 开发时间：2025/12/22 14:18
# _*_coding:UTF-8_*_
# model.py
# DeBERTa-v3-base sentiment classification model

import torch
import torch.nn as nn
from transformers import AutoModel

class SentimentModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(cls_output)
        return logits

