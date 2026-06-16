import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import entropy
from tqdm import tqdm
from torchmetrics.image.inception import InceptionScore
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

class InceptionScoreCalculator:
    def __init__(self, device, batch_size=64, splits=10, feature='logits_unbiased'):
        """
        初始化 IS 计算器
        
        Args:
            device: 运行设备
            batch_size: 计算时的批次大小 (防止显存溢出)
            splits: IS 计算的切分数 (标准为 10)
            feature: Inception 特征层 ('logits_unbiased' 是推荐的现代标准)
        """
        self.device = device
        self.batch_size = batch_size
        self.metric = InceptionScore(feature=feature, splits=splits).to(device)

    def compute_inception_score(self, images, reset=True):
        """
        计算给定 Tensor 的 Inception Score
        
        Args:
            images (torch.Tensor): 形状为 (N, 3, H, W) 的完整图像张量
                                   数值范围预期在 [-1, 1] 之间 (生成模型常用输出)
            reset (bool): 计算前是否重置内部状态 (默认 True)
        
        Returns:
            tuple: (is_mean, is_std)
        """
        if reset:
            self.metric.reset()
            
        num_images = images.size(0)
        batches = torch.split(images, self.batch_size)
        
        for batch in tqdm(batches, desc="IS Processing"):
            # 将当前批次移动到 GPU
            batch = batch.to(self.device)
                        
            batch = torch.clamp(batch, -1.0, 1.0)
            batch_uint8 = ((batch + 1) / 2.0 * 255).to(torch.uint8)
            self.metric.update(batch_uint8)
            del batch, batch_uint8

        # 3. 计算最终结果
        is_mean, is_std = self.metric.compute()
        
        return is_mean.item(), is_std.item()
