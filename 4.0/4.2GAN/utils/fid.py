import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
from scipy.stats import entropy
from torchmetrics.image.inception import InceptionScore
from pytorch_fid.inception import InceptionV3
import torch.nn.functional as F
from pytorch_fid.fid_score import calculate_frechet_distance

def fid_calculator(real_images, fake_images, batch_size=50, device='cuda'):
    """
    计算 FID。
    real_images, fake_images: Tensor [N, 3, H, W] (建议在 CPU 上以节省显存)
    """
    # 加载 pytorch-fid 官方 InceptionV3 (Block 3, 2048-dim)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    def get_inception_stats(images):
        """分批计算统计量 (Mean, Cov)"""
        act_list = []
        n_batches = (len(images) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                # 1. 从 CPU 取出一个 batch 放到 GPU
                batch = images[i * batch_size : (i + 1) * batch_size].to(device)
                
                # 2. 预处理: [-1, 1] -> [0, 1]
                if batch.min() < 0:
                    batch = (batch + 1) / 2.0
                
                # 3. Resize 到 299x299 (Inception 标准)
                if batch.shape[2] != 299 or batch.shape[3] != 299:
                    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                
                # 4. 前向传播
                pred = model(batch)[0]
                
                # 5. Global Pooling
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
                
                act_list.append(pred.squeeze(3).squeeze(2).cpu().numpy())

        act = np.concatenate(act_list, axis=0)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    print("Computing stats for Real images...")
    m1, s1 = get_inception_stats(real_images)
    
    print("Computing stats for Fake images...")
    m2, s2 = get_inception_stats(fake_images)
    
    print("Computing Frechet Distance...")
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value
