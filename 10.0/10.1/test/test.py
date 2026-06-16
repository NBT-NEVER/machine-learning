# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：test.py
# 开发时间：2025/12/8

import os
import sys
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

# 切换到当前脚本路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Working dir:", os.getcwd())
print("Files:", os.listdir())

# 导入 multi_model_benchmark
sys.path.append("..")
from multi_model_benchmark import run_multi_model_benchmark


#############################################
#            读取 CSV （安全版）
#############################################
def load_csv(path):
    """
    自动读取以下格式：
    id, content, label
    或
    id, text, label
    或
    content, label
    """

    df = pd.read_csv(path)

    # 自动识别文本列
    text_col = None
    for col in df.columns:
        if col.lower() in ["content", "text", "review", "sentence"]:
            text_col = col
            break
    if text_col is None:
        raise ValueError("CSV 中未找到文本列（content/text/...）")

    # 自动识别标签列
    label_col = None
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break
    if label_col is None:
        raise ValueError("CSV 中未找到 label 列")

    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[label_col].astype(int).tolist()

    return texts, labels


#############################################
#        主流程（采用 stratify 正确划分）
#############################################
if __name__ == "__main__":
    print("\n===== Loading dataset =====")

    train_path = "train.csv"

    # 读取完整版训练数据
    texts, labels = load_csv(train_path)

    print(f"Total samples loaded: {len(texts)}")

    # ======= 正确的划分方式，与 BERT 代码完全一致 =======
    train_texts, val_texts, train_labels, val_labels =train_test_split(
            texts, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels     # ★★★★★ 必须 stratify，防止类别失衡
        )

    print(f"Train set: {len(train_texts)} samples")
    print(f"Val   set: {len(val_texts)} samples")

    # ===== 开始多模型基准测试 =====
    print("\n===== Start Benchmark =====\n")

    run_multi_model_benchmark(
        train_texts, train_labels,
        val_texts, val_labels
    )
