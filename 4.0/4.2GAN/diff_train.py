import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from multiprocessing import freeze_support
from utils.unet import SimpleUNet



def main():

    class Config:
        # 核心训练参数
        batch_size = 128         # input batch size for training
        epochs = 20000             # number of epochs to train
        lr = 0.0002              # learning rate
        beta1 = 0.5              # Adam beta1
        
        # Diffusion 参数
        num_timesteps = 1000     # number of diffusion timesteps
        beta_start = 0.0001      # start value of beta for noise schedule
        beta_end = 0.02          # end value of beta for noise schedule

        # 系统与保存参数
        no_cuda = False          # disables CUDA training
        seed = 42                # random seed
        save_model = True        # For Saving the current Model
        load_model = './best_diffusion_model.pth'       # Path to load model for resuming training
        save_interval = 10       # Interval to save model checkpoints

    # 实例化配置
    args = Config()

    # Check if CUDA is available
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Initialize model, optimizer, loss function
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    criterion = nn.MSELoss()

    # Define noise schedule (linear beta schedule)
    betas = torch.linspace(args.beta_start, args.beta_end, args.num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

    # Load model if specified
    start_epoch = 0
    if args.load_model:
        if os.path.isfile(args.load_model):
            checkpoint = torch.load(args.load_model, map_location=device, weights_only=False)
            # checkpoint = torch.load(path)
            # args_dict = checkpoint['args']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint '{args.load_model}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.load_model}'")

    # Function to sample timesteps
    def sample_timesteps(batch_size):
        return torch.randint(0, args.num_timesteps, (batch_size,), device=device)

    # Function to add noise to images
    def add_noise(images, t):
        # Generate random noise
        noise = torch.randn_like(images, device=device)

        # Compute alpha_cumprod for each timestep in the batch
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[t])[:, None, None, None]

        # Add noise to images
        noisy_images = sqrt_alphas_cumprod * images + sqrt_one_minus_alphas_cumprod * noise

        return noisy_images, noise

    # Function to train one epoch
    def train_one_epoch(epoch):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            batch_size = images.size(0)

            # Sample timesteps
            t = sample_timesteps(batch_size)

            # Add noise to images
            noisy_images, noise = add_noise(images, t)

            # Forward pass
            optimizer.zero_grad()
            predicted_noise = model(noisy_images, t)

            # Calculate loss (MSE between predicted noise and actual noise)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)
        return epoch_loss

    # Function to generate and save samples during training
    def generate_samples(epoch, model, betas, alphas_cumprod, alphas_cumprod_prev, num_samples=64, save_dir='./diff_samples'):
        """
        DDPM 标准采样流程
        """
        model.eval()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 计算采样过程中需要的系数
        # 注意：这里的 alphas 是每一步的 alpha，不是累乘
        alphas = 1.0 - betas 
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        with torch.no_grad():
            # 1. 从纯高斯噪声开始 x_T
            x = torch.randn(num_samples, 3, 32, 32, device=device)

            # 2. 倒序遍历: T-1, ..., 0
            for i in tqdm(reversed(range(0, len(betas))), desc="Sampling", leave=False):
                t = torch.full((num_samples,), i, device=device, dtype=torch.long)

                # 预测噪声
                noise_pred = model(x, t)

                # 提取当前步的系数
                beta_t = betas[i]
                sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
                sqrt_recip_alpha_t = sqrt_recip_alphas[i]

                # 计算均值 mean
                # mean = 1/sqrt(alpha) * (x - (beta / sqrt(1-alpha_bar)) * noise_pred)
                mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred)

                # 如果不是最后一步 (t=0)，则加入随机噪声
                if i > 0:
                    noise = torch.randn_like(x)
                    # 方差可以用 beta_t 或者计算出来的 posterior_variance
                    # 这里使用最简单的 sigma = sqrt(beta_t) 也就是论文中的 Option 1
                    sigma = torch.sqrt(beta_t) 
                    x = mean + sigma * noise
                else:
                    x = mean

        # 保存图片
        save_path = os.path.join(save_dir, f'epoch_{epoch+1}.png')
        vutils.save_image(x, save_path, normalize=True, nrow=8) # normalize=True会自动将[-1,1]映射到[0,1]
        print(f"Saved samples to {save_path}")
        model.train()

    # Main training loop
    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        loss = train_one_epoch(epoch)

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss:.4f}")

        # Generate and save samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            generate_samples(epoch, model, betas, alphas_cumprod, alphas_cumprod_prev)

        # Save model if it's the best so far
        if loss < best_loss and args.save_model:
            best_loss = loss
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                # 'args': args,
            }, f'./best_diffusion_model.pth')
            print(f"Best model saved with loss: {best_loss:.4f}")

        # Save model at intervals
        if (epoch + 1) % args.save_interval == 0 and args.save_model:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                # 'args': args,
            }, f'./diffusion_model_epoch_{epoch+1}.pth')

    print("Training completed!")

if __name__ == '__main__':
    freeze_support()
    main()