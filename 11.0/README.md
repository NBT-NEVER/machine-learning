# 11.0 实验说明

`11.0` 目录对应情感五分类实验，整体思路是在共享数据集上对比不同大型预训练语言模型的效果，并保留训练日志、推理脚本和实验输出。

## 共享数据

- `data/train.csv`、`data/test.csv`、`data/sample_submission.csv` 是两个子实验共同使用的数据文件。
- `data/glove.6B.300d.txt` 一并保留，供需要传统词向量初始化时参考。

## 子实验组成

### 11.1 RoBERTa-large 版本

- `train.py` 使用 `roberta-large` 进行五分类训练。
- `model.py` 定义标准分类头，`model_small.py` 保留了一个带 dropout 的轻量替代实现。
- `submission.py` 用训练好的模型生成提交文件。
- 原始记录里给出了一个伪测试精度：`0.66920`。

### 11.2 DeBERTa-v3-base / large 版本

- `train.py` 使用 `microsoft/deberta-v3-base` 进行五分类训练。
- `model.py` 定义编码器与线性分类头。
- `submission.py` 根据训练配置导出提交结果。
- 原始记录中保留了 `0.66940` 的伪测试精度，以及若干轮训练的 `train_loss / val_loss / val_acc`。

## 目录结构

```text
11.0/
├─11.1/
│ ├─train.py
│ ├─model.py
│ ├─model_small.py
│ ├─submission.py
│ └─out/
├─11.2/
│ ├─train.py
│ ├─model.py
│ ├─submission.py
│ └─out/
└─data/
  ├─train.csv
  ├─test.csv
  ├─sample_submission.csv
  └─glove.6B.300d.txt
```

## 托管说明

- 训练日志、曲线图、样例输出与数据文件继续保留。
- 最佳模型权重文件不纳入仓库，但可通过现有训练脚本重新生成。
