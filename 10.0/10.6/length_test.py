# -*- coding: utf-8 -*-
# 开发人员：NBT
# 文件名称：length_analysis.py
# 功能：评论长度统计 + 图形 + max_len 推荐

import pandas as pd
import matplotlib.pyplot as plt

# ================================
# 1. 读取数据
# ================================
df = pd.read_csv("../data/train.csv")

# 按“词数”统计长度
lengths = df["content"].fillna("").astype(str).apply(lambda x: len(x.split()))

print("========== 统计信息 ==========")
print("平均长度:", lengths.mean())
print("最长评论:", lengths.max())
print("90% 评论长度:", lengths.quantile(0.9))
print("95% 评论长度:", lengths.quantile(0.95))
print("99% 评论长度:", lengths.quantile(0.99))


# ================================
# 2. 推荐 max_len
# ———— 推荐使用 90% 或 95% 分位
# ================================
len_90 = int(lengths.quantile(0.9))
len_95 = int(lengths.quantile(0.95))

print("\n========== 推荐 max_len ==========")
print(f"建议 max_len（90%长度）：{len_90}")
print(f"建议 max_len（95%长度）：{len_95}")
print("说明：")
print("- 90%：更快训练，减少 padding，提高效率")
print("- 95%：更稳健，适合追求 accuracy")


# ================================
# 3. 绘制长度直方图 Histogram
# ================================
plt.figure(figsize=(8,4))
plt.hist(lengths, bins=50)
plt.xlabel("lengths")
plt.ylabel("frequency")
plt.title("lengths")
plt.tight_layout()
plt.savefig("./out/length_histogram.png")
print("\nSaved: ./out/length_histogram.png")


# ================================
# 4. 绘制折线图（CDF 累积分布）
#    显示有多少评论长度落在前 X 词以内
# ================================
sorted_len = lengths.sort_values().reset_index(drop=True)
cdf = sorted_len.index / len(sorted_len)

plt.figure(figsize=(8,4))
plt.plot(sorted_len, cdf)
plt.xlabel("len")
plt.ylabel("rate")
plt.title("CDF")
plt.grid(True)
plt.tight_layout()
plt.savefig("./out/length_cdf.png")
print("Saved: ./out/length_cdf.png")



