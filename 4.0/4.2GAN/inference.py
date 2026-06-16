#inference.py用于生成图片
import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.generator import Generator
from utils.fid import fid_calculator
from torchvision import datasets
from utils.inception_score import InceptionScoreCalculator

# --- 1. 更新后的推理配置参数 ---
class InferenceConfig:
    # 路径与文件
    model_path = './gan_model_epoch_50.pth'
    output_dir = './output'
    output_name = 'gan_generated_epoch_50.png'
    
    # 模型参数
    latent_dim = 100
    ngf = 64
    
    # 生成参数
    num_images = 100
    seed = 30
    no_cuda = False

# 实例化配置
args = InferenceConfig()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 主推理逻辑 (保持不变) ---
def run_inference():
    # 检查设备
    print(f"Running inference on: {device}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # 初始化生成器
    try:
        # 确保 Generator 类已定义
        netG = Generator(latent_dim=args.latent_dim, ngf=args.ngf).to(device)
    except NameError:
        print("Error: 'Generator' class is not defined. Please run the cell defining the Generator model first.")
        return

    # 加载模型权重
    if os.path.isfile(args.model_path):
        print(f"Loading model from {args.model_path}...")
        try:
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            # 兼容处理
            if isinstance(checkpoint, dict) and 'generator' in checkpoint:
                netG.load_state_dict(checkpoint['generator'])
            else:
                netG.load_state_dict(checkpoint)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Error: No model found at {args.model_path}")
        return

    # 设置为评估模式
    netG.eval()
    
    # 生成随机噪声 (Batch, 100, 1, 1)
    noise = torch.randn(args.num_images, args.latent_dim, 1, 1, device=device)
    fake_images_list = []
    
    # 生成图片
    print("Generating images...")
    with torch.no_grad():
        # 使用 range 步长进行循环
        for i in tqdm(range(0, args.num_images, 32), desc="Generating"):
            current_batch_size = min(32, args.num_images - i)
            noise = torch.randn(current_batch_size, args.latent_dim, 1, 1, device=device)
            fake_batch = netG(noise)
            fake_images_list.append(fake_batch)
    fake_images = torch.cat(fake_images_list, dim=0)

    print("Calculating FID...")
    random.seed(args.seed)
    real_indices = random.sample(range(len(test_dataset)), args.num_images)
    real_images = torch.stack([test_dataset[i][0] for i in real_indices]).to(device)
    
    # Calculate FID score
    fid_score = fid_calculator(real_images, fake_images, batch_size=32, device=device)
    print(f"FID Score: {fid_score:.4f}")

    print("Calculating Inception Score...")
    is_calculator = InceptionScoreCalculator(device=device)
    inception_score, std_inception_score = is_calculator.compute_inception_score(
        fake_images
    )
    print(f"Inception Score: {inception_score:.4f} ± {std_inception_score:.4f}")
        
    # --- 保存图片 ---
    output_path = os.path.join(args.output_dir, args.output_name)
    fake_images_output = fake_images[:64]
    vutils.save_image(fake_images_output, output_path, normalize=True, nrow=8)
    print(f"Images saved to: {output_path}")
    
    # --- 可视化显示 ---
    print("\nDisplaying generated results:")
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Seed: {args.seed} | File: {args.output_name}")
    grid_img = vutils.make_grid(fake_images.cpu()[:64], padding=2, normalize=True, nrow=8)
    plt.imshow(np.transpose(grid_img, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    run_inference()