# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：Statistics.py
# 开发时间：2025/12/24 16:22
# 功能：分析多个 submission.csv 的预测分歧，并生成融合结果

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

OUT_DIR = "./out"
SAVE_FUSED = "./out/submission.csv"


def load_submissions(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "*.csv")))
    if len(files) < 2:
        raise ValueError("out 目录下至少需要 2 个 submission.csv")

    print("=== 发现的 submission 文件 ===")
    for f in files:
        print(os.path.basename(f))

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = df.sort_values("id").reset_index(drop=True)
        dfs.append(df["label"].values)

    ids = pd.read_csv(files[0])["id"].values
    preds = np.vstack(dfs)  # shape: [num_models, num_samples]

    return ids, preds, files


def compute_statistics(preds):
    """
    preds: [num_models, num_samples]
    """
    num_models = preds.shape[0]

    vote_sum = preds.sum(axis=0)
    vote_ratio = vote_sum / num_models

    # 是否存在分歧
    disagree = (vote_sum != 0) & (vote_sum != num_models)

    # 熵（衡量不确定性）
    eps = 1e-9
    p = vote_ratio
    entropy = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))

    stats = pd.DataFrame({
        "vote_sum": vote_sum,
        "vote_ratio": vote_ratio,
        "disagree": disagree,
        "entropy": entropy
    })

    return stats


def plot_statistics(stats):
    plt.figure(figsize=(12, 4))

    # 1. 投票比例分布
    plt.subplot(1, 2, 1)
    plt.hist(stats["vote_ratio"], bins=20)
    plt.xlabel("Vote Ratio (predict=1)")
    plt.ylabel("Count")
    plt.title("Prediction Agreement Distribution")

    # 2. 不一致样本比例
    plt.subplot(1, 2, 2)
    counts = Counter(stats["disagree"])
    plt.bar(["Agree", "Disagree"], [counts[False], counts[True]])
    plt.title("Agreement vs Disagreement Samples")

    plt.tight_layout()
    plt.show()


def generate_fused_submission(ids, preds):
    """
    简单多数投票
    """
    vote_sum = preds.sum(axis=0)
    fused_label = (vote_sum >= (preds.shape[0] / 2)).astype(int)

    fused = pd.DataFrame({
        "id": ids,
        "label": fused_label
    })

    fused.to_csv(SAVE_FUSED, index=False)
    print(f"\n融合后的 submission 已保存到：{SAVE_FUSED}")


def main():
    ids, preds, files = load_submissions(OUT_DIR)

    print(f"\n模型数量: {preds.shape[0]}")
    print(f"测试样本数: {preds.shape[1]}")

    stats = compute_statistics(preds)

    # 关键统计信息
    disagree_ratio = stats["disagree"].mean()
    print(f"\n存在预测分歧的样本比例: {disagree_ratio:.2%}")

    print("\n预测最不稳定的样本（Top 10 by entropy）:")
    print(stats.sort_values("entropy", ascending=False).head(10))

    plot_statistics(stats)
    generate_fused_submission(ids, preds)


if __name__ == "__main__":
    main()
