# 2.0 实验说明

`2.0` 目录对应医学图像二分类实验，当前主要内容集中在 `2.3` 版本。该实验以 `2-MedImage-TrainSet` 和 `2-MedImage-TestSet` 为基础，对不同网络结构进行训练与比较。

## 实验内容

### 2.3.1 EfficientNet-B0 基线

- `ml-course2.3.1.py` 使用 `torchvision.models.efficientnet_b0` 构建二分类模型。
- 脚本中包含数据增强、训练循环、验证、最佳模型保存等标准流程。
- 适合作为医学图像二分类的起始版本。

### 2.3.2 自定义网络版本

- `ml-course2.3.2.py` 在同一任务上尝试自定义网络结构，代码中可见 Inception 模块设计。
- 这一版体现了“在课程给定任务上自行改结构”的思路。

### 2.3.3 EfficientNet-B4 强化版本

- `ml-course2.3.3.py` 将主干网络替换为 `efficientnet_b4`。
- 该版本继续沿用同一训练流程，但模型容量更大，目标是提升分类效果。

## 数据与附加材料

- `2-MedImage-TrainSet.zip` 和 `2-MedImage-TestSet.zip` 保留原始压缩包，便于重新解压实验数据。
- `ml-course2.ipynb` 是 notebook 形式的课程实验文件。
- `loss_curve.png`、`roc_curve.png` 保留了训练结果可视化。
- `使用方法.pdf` 是原始实验说明材料。

## 目录结构

```text
2.0/
└─2.3/
  ├─ml-course2.3.1.py
  ├─ml-course2.3.2.py
  ├─ml-course2.3.3.py
  ├─ml-course2.ipynb
  ├─2-MedImage-TrainSet.zip
  ├─2-MedImage-TestSet.zip
  └─使用方法.pdf
```

## 运行提示

- 三个主脚本都默认在当前目录下寻找训练集和测试集。
- 脚本里保留了早期本地路径切换语句，迁移到新环境时需要先检查并按当前机器目录调整。
- 仓库中不再保留训练得到的最佳权重文件，但训练曲线和数据集文件会继续保留。
