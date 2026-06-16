# 3.0 实验说明

`3.0` 目录对应显著性检测实验，核心任务是根据输入彩色图像预测显著性图，并使用 CC、KL 等指标进行评估。

## 子实验组成

### 3.1 ResNet18 显著性检测主实验

- `main.py` 负责训练流程，使用 `ResNet18Saliency` 网络进行显著性图预测。
- `model.py` 定义了编码器-解码器结构，编码端使用 ResNet18，多层特征通过反卷积逐级恢复分辨率。
- `dataset.py` 按 `Stimuli/` 与 `FIXATIONMAPS/` 的目录结构组织样本和显著性标注。
- `metric.py` 给出了 CC 和 KL 散度等评估指标实现。
- `test.py` 负责整体测试与结果保存，`test_category.py` 负责按类别统计评估结果。

### 3.2 notebook 版本

- `ml-course3.ipynb` 与 `ml-course3.py` 保留了该实验的 notebook 和脚本化版本，适合回看课程过程。

## 数据与结果

- `3-Saliency-TrainSet.zip`、`3-Saliency-TestSet.zip` 保留训练集与测试集压缩包。
- `category.zip` 提供类别信息或辅助文件。
- `README.pdf` 是原始实验说明文档。
- 训练得到的最佳模型权重不纳入仓库，但评估脚本、数据与说明完整保留。

## 推荐阅读顺序

1. 先看 `main.py`，了解训练参数、数据集路径和保存逻辑。
2. 再看 `model.py` 与 `dataset.py`，理解网络结构和样本组织方式。
3. 最后查看 `test.py`、`test_category.py` 和 `metric.py`，了解评估过程。

## 目录结构

```text
3.0/
├─3.1/
│ ├─main.py
│ ├─model.py
│ ├─dataset.py
│ ├─metric.py
│ ├─test.py
│ ├─test_category.py
│ ├─3-Saliency-TrainSet.zip
│ └─3-Saliency-TestSet.zip
└─3.2/
  ├─ml-course3.ipynb
  └─ml-course3.py
```
