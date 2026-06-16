# 1.0 实验说明

`1.0` 目录围绕 MNIST 手写数字识别展开，包含三组逐步扩展的实验：基础分类脚本、单文件 PyTorch 识别程序，以及带图形界面的卷积网络演示项目。

## 子实验组成

### 1.1 基础入门版本

- `main.py` 使用 TensorFlow/Keras 完成一个简单的全连接分类基线。
- `train.py` 与 `train_iter.py` 使用 `cnn.py` 中定义的卷积网络进行 PyTorch 训练与测试。
- 这一部分适合快速熟悉 MNIST 数据加载、训练循环和测试集评估流程。

### 1.2 单文件 PyTorch 识别脚本

- `cnn_mnist_pytorch.py` 集中完成数据加载、网络定义、训练、推理与可视化。
- `MNIST/raw` 与 `MNIST/processed` 内保留了原始和处理后的数据文件，便于直接运行。
- 该版本强调“一个脚本跑通训练和识别”的实验形式。

### 1.3 图形界面识别版本

- `mnist_cnn_gui_main.py` 是项目入口，结合 `qt/` 下的界面文件和画板组件实现手写数字 GUI 识别。
- `simple_convnet.py`、`deep_convnet.py`、`common/` 构成 NumPy/CuPy 风格的卷积网络实现。
- `train_convnet.py` 与 `train_deepnet.py` 用于训练简单卷积网络和深层卷积网络。
- `image/mnist_gui.png` 给出了界面示意，`流程指导.txt` 保留了原有的环境与运行提示。

## 目录结构

```text
1.0/
├─1.1/  基础训练脚本
├─1.2/  单文件 PyTorch 版本 + MNIST 数据
└─1.3/  GUI 识别版本
  ├─common/
  ├─image/
  └─qt/
```

## 运行建议

- `1.1` 适合先看整体训练流程。
- `1.2` 适合查看一个完整脚本如何串起训练与推理。
- `1.3` 适合演示卷积网络识别结果和界面交互。

如果直接运行 `1.3`，建议先按该目录中的依赖文件安装环境，再准备好 MNIST 数据集相关目录。
