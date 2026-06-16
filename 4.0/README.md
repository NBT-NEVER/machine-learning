# 4.0 实验说明

`4.0` 目录聚焦生成模型实验，主题是基于 CIFAR-10 数据集进行图像生成。当前主要代码集中在 `4.2GAN`，同时保留了 `4.1` 的 notebook 记录。

## 实验组成

### 4.1 notebook 草稿

- `ml-course4.ipynb` 记录了该阶段的实验过程，适合作为课程草稿或过程存档查看。

### 4.2GAN 主实验目录

#### GAN 分支

- `train.py` 负责 GAN 训练。
- `inference.py` 负责加载生成器并输出生成图片，同时计算 FID 和 Inception Score。
- `utils/generator.py`、`utils/discriminator.py` 提供生成器和判别器结构定义。

#### Diffusion 分支

- `diff_train.py` 负责扩散模型训练。
- `diff_inference.py` 用于扩散模型推理与结果保存。
- `diff_viz.py` 用于可视化扩散过程或生成结果。
- `utils/unet.py` 是扩散模型中的 U-Net 主体实现。

#### 评估与辅助工具

- `utils/fid.py`、`utils/inception_score.py` 用于生成质量评估。
- `utils/save_cifar_images.py` 用于数据或样例图整理。

## 数据与结果目录

- `data/` 中保留 CIFAR-10 原始压缩包和解压后的批文件。
- `cifar_images.zip` 是辅助图像资源。
- `output/`、`output_diff/` 和 `samples/` 保存了不同训练阶段的生成结果图。
- 训练得到的权重文件不纳入仓库，但数据、脚本和输出样图会继续保留。

## 目录结构

```text
4.0/
├─4.1/
│ └─ml-course4.ipynb
└─4.2GAN/
  ├─train.py
  ├─inference.py
  ├─diff_train.py
  ├─diff_inference.py
  ├─diff_viz.py
  ├─utils/
  ├─data/
  ├─output/
  ├─output_diff/
  └─samples/
```

## 使用建议

- 如果想看传统 GAN 流程，优先阅读 `train.py` 与 `inference.py`。
- 如果想看扩散模型流程，优先阅读 `diff_train.py` 与 `diff_inference.py`。
- 运行前建议确认 `data/`、输出目录和设备配置是否符合当前环境。
