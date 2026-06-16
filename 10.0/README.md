# 10.0 实验说明

`10.0` 目录集中保存情感二分类实验，是整个仓库中文本方向内容最完整的一组。实验从传统机器学习方法逐步扩展到 GloVe + LSTM、RoBERTa 和 DeBERTa 等预训练模型。

## 共享数据目录

- `data/train.csv`、`data/test.csv`、`data/sample_submission.csv` 是多个子实验共用的数据文件。
- `data/glove.6B.300d.txt` 是 GloVe 词向量文件，体积较大，仓库中通过 Git LFS 管理。

## 子实验概览

### 10.1 BERT / RoBERTa 规范化训练脚本

- `sentiment_analysis_experiments.py` 是二分类训练主脚本，包含训练验证划分、日志记录、最佳模型保存、损失曲线绘制和测试集预测。
- `submission.py` 用训练得到的最佳模型生成 `submission.csv`。
- `test/` 目录中保留了多模型基准测试脚本与对比结果图。

### 10.2 传统机器学习与早期深度学习对比

- `code/` 目录提供了 baseline、Naive Bayes、Logistic Regression、Decision Tree、Random Forest、SVM、XGBoost、LSTM、CNN 和多数投票集成等方法。
- `dataset/positive-words.txt` 与 `dataset/negative-words.txt` 用于词典特征。
- `docs/` 中保留了项目报告与绘图 notebook。

### 10.3 非预训练 GloVe + LSTM 版本

- `train.py`、`model.py`、`submission.py` 构成一个不依赖大语言模型的英文情感分类流程。
- 该版本额外保留了 `glove_test.py`、`length_test.py`，用于词向量与长度分布分析。
- `out/` 和 `saved_models/` 中保留了日志、参数和结果图，训练权重文件已从托管范围中排除。

### 10.4 稳定收敛版本

- `train_p.py` 是在前一版本基础上的稳定训练脚本。
- `model2.py` 定义了 Embedding + 双层 BiLSTM + 轻量 Attention + Dropout + 全连接层的结构。
- 这一版更强调训练稳定性和结构清晰性。

### 10.5 预训练模型二分类版本

- `train.py`、`model.py`、`submission.py` 使用统一的 `TRAIN_CONFIG` 驱动训练与推理。
- 代码中配置为 `roberta-base`，但实验说明保留了 DeBERTa-v3-base 的尝试痕迹。
- `saved_models/` 中保留训练日志与损失曲线，最佳权重不纳入仓库。

### 10.6 RoBERTa-base + Focal Loss 版本

- `train.py` 在标准 transformer 二分类流程上加入类别权重和 Focal Loss。
- `model.py` 使用 `roberta-base` 编码器与线性分类头。
- `submission.py` 根据 `TRAIN_CONFIG` 自动读取参数与输出文件名。
- `out/`、`out_deep/` 和 `saved_models/` 中保留了多组提交文件、日志和结果图。

## 建议阅读顺序

1. 想看完整的传统方法对比，可先读 `10.2`。
2. 想看现代 transformer 二分类流程，可读 `10.1`、`10.5`、`10.6`。
3. 想看不依赖预训练模型的实现，可读 `10.3` 和 `10.4`。

## 托管说明

- 数据文件、报告、日志、曲线图和提交结果会保留。
- 训练得到的最佳模型权重、预训练权重缓存以及 HuggingFace 下载缓存不提交。
