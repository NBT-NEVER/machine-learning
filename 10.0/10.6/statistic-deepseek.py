# _*_coding:UTF-8_*_
# 功能：7 个 submission 融合 + DeepSeek 裁决分歧样本（支持断点）

import os
import glob
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# ================= 配置 =================
READ_DIR = "./out"
OUT_DIR = "./out_deep"
SAVE_FUSED = "./out_deep/submission_fused_deepseek.csv"
CHECKPOINT_FILE = "./out_deep/deepseek_checkpoint.csv"
BATCH_SIZE = 100

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ================= 读取 submission =================
def load_submissions():
    files = sorted(glob.glob(os.path.join(READ_DIR, "*.csv")))
    if len(files) < 2:
        raise RuntimeError("submission 文件不足")

    dfs = [pd.read_csv(f).sort_values("id").reset_index(drop=True) for f in files]
    preds = np.vstack([df["label"].values for df in dfs])
    ids = dfs[0]["id"].values

    return ids, preds

# ================= DeepSeek =================
def deepseek_predict(texts):
    results = []
    for t in texts:
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "以影评情感分析为背景，你需要对文本情感态度进行二分类，分辨其属于正面（反馈为1）或负面（反馈为0）态度 "
                        "Classify this sentence as positive or negative.Return only ONE digit:\n"
                        "1 = positive sentiment\n"
                        "0 = negative sentiment"
                    )
                },
                {"role": "user", "content": t}
            ],
            stream=False
        )
        results.append(int(r.choices[0].message.content.strip()))
    return results

# ================= 主逻辑 =================
def main():
    ids, preds = load_submissions()
    num_models = preds.shape[0]

    # 多数投票作为初始结果
    final_labels = (preds.sum(axis=0) >= (num_models / 2)).astype(int)

    # 找分歧样本
    vote_sum = preds.sum(axis=0)
    disagree_idx = np.where((vote_sum != 0) & (vote_sum != num_models))[0]

    print(f"分歧样本数: {len(disagree_idx)}")

    # 断点
    start = 0
    if os.path.exists(CHECKPOINT_FILE):
        start = int(pd.read_csv(CHECKPOINT_FILE)["processed"].iloc[0])
        print(f"从断点继续：{start}")

    df_test = pd.read_csv("../data/test.csv")

    # ===== 分批处理 =====
    for i in tqdm(range(start, len(disagree_idx), BATCH_SIZE)):
        batch_idx = disagree_idx[i:i + BATCH_SIZE]
        texts = df_test.iloc[batch_idx]["content"].astype(str).tolist()

        preds_ds = deepseek_predict(texts)

        # 用 DeepSeek 覆盖最终预测
        for idx, p in zip(batch_idx, preds_ds):
            final_labels[idx] = p

        # 保存融合结果
        pd.DataFrame({
            "id": ids,
            "label": final_labels
        }).to_csv(SAVE_FUSED, index=False)

        # 保存断点
        pd.DataFrame({"processed": [i + len(batch_idx)]}).to_csv(
            CHECKPOINT_FILE, index=False
        )

        print(f"已处理 {i + len(batch_idx)} / {len(disagree_idx)}")

    print("DeepSeek 融合完成")
    print(f"结果文件：{SAVE_FUSED}")

# ================= 入口 =================
if __name__ == "__main__":
    main()
