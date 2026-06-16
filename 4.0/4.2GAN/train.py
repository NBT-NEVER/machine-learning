#train.py用于训练GAN网络模型
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from multiprocessing import freeze_support
from utils.generator import Generator
from utils.discriminator import Discriminator
from utils.fid import fid_calculator


def main():

    class Config:
        # 核心训练参数
        epochs = 20000             # 训练轮数
        batch_size = 64          # 批次大小
        lr = 1e-4                # 学习率 (0.0001)
        lr_decay = 0.9           # 学习率衰减率
        decay_epochs = 20        # 每多少轮衰减一次
        beta1 = 0.5              # Adam优化器的beta1
        # 模型结构参数
        latent_dim = 100         # 噪声维度
        ngf = 64                 # 生成器特征图基数
        ndf = 64                 # 判别器特征图基数
        # 系统与保存参数
        seed = 42                # 随机种子
        save_interval = 1        # 每多少轮保存一次模型
        no_cuda = False          # 是否禁用CUDA
        save_model = True        # 是否保存模型
        load_model = './best_gan_model.pth'                 # 加载预训练模型的路径 (如 './gan_model_epoch_10.pth')
    
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
    
    # Initialize models
    netG = Generator(latent_dim=args.latent_dim, ngf=args.ngf).to(device)
    netD = Discriminator(ndf=args.ndf).to(device)
    
    # Initialize optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
    # Initialize learning rate schedulers
    # StepLR: decay learning rate by a factor of lr_decay every decay_epochs epochs
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=args.decay_epochs, gamma=args.lr_decay)
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=args.decay_epochs, gamma=args.lr_decay)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Initialize labels
    real_label = 1.0
    fake_label = 0.0
    
    # Initialize FID calculator
    print("Initializing FID calculator...")
    
    # Load model if specified
    start_epoch = 0
    best_fid = float('inf')  # FID is better when it's lower
    
    if args.load_model:
        if os.path.isfile(args.load_model):
            checkpoint = torch.load(args.load_model, map_location=device, weights_only=False)
            netG.load_state_dict(checkpoint['generator'])
            netD.load_state_dict(checkpoint['discriminator'])
            optimizerG.load_state_dict(checkpoint['optimizerG'])
            optimizerD.load_state_dict(checkpoint['optimizerD'])
            # Load scheduler states if they exist in the checkpoint
            if 'schedulerG' in checkpoint and 'schedulerD' in checkpoint:
                schedulerG.load_state_dict(checkpoint['schedulerG'])
                schedulerD.load_state_dict(checkpoint['schedulerD'])
                print("Loaded scheduler states from checkpoint")
            start_epoch = checkpoint['epoch']
            if 'best_fid' in checkpoint:
                # best_fid = checkpoint['best_fid']
                print(f"Loaded checkpoint '{args.load_model}' (epoch {start_epoch}, best FID: {best_fid:.4f})")
            else:
                print(f"Loaded checkpoint '{args.load_model}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.load_model}'")
    
    # Function to train one epoch
    def train_one_epoch(epoch):
        netG.train()
        netD.train()
        running_loss_g = 0.0
        running_loss_d = 0.0
        
        # Create separate progress bars for discriminator and generator
        total_batches = len(train_loader)
        pbar_d = tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{args.epochs} - Discriminator")
        pbar_g = tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{args.epochs} - Generator")
        
        for i, (real_images, _) in enumerate(train_loader):
            # Move real images to device
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Train with real images
            netD.zero_grad()
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_images).view(-1)
            loss_d_real = criterion(output, label)
            loss_d_real.backward()
            D_x = output.mean().item()
            
            # Train with fake images
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_images.detach()).view(-1)
            loss_d_fake = criterion(output, label)
            loss_d_fake.backward()
            D_G_z1 = output.mean().item()
            
            # Update discriminator
            loss_d = loss_d_real + loss_d_fake
            optimizerD.step()
            
            # Update discriminator progress bar
            running_loss_d += loss_d.item()
            avg_loss_d = running_loss_d / (i + 1)
            pbar_d.update(1)
            pbar_d.set_postfix({
                'Loss D': f'{loss_d.item():.4f}',
                'Avg Loss D': f'{avg_loss_d:.4f}',
                'D(x)': f'{D_x:.4f}',
                'D(G(z))': f'{D_G_z1:.4f}'
            })
            
            # ---------------------
            #  Train Generator
            # ---------------------
            netG.zero_grad()
            label.fill_(real_label)  # Fake labels are real for generator cost
            output = netD(fake_images).view(-1)
            loss_g = criterion(output, label)
            loss_g.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # Update generator progress bar
            running_loss_g += loss_g.item()
            avg_loss_g = running_loss_g / (i + 1)
            pbar_g.update(1)
            pbar_g.set_postfix({
                'Loss G': f'{loss_g.item():.4f}',
                'Avg Loss G': f'{avg_loss_g:.4f}',
                'D(G(z))': f'{D_G_z2:.4f}'
            })
        
        # Close progress bars
        pbar_d.close()
        pbar_g.close()
        
        epoch_loss_g = running_loss_g / total_batches
        epoch_loss_d = running_loss_d / total_batches
        
        return epoch_loss_g, epoch_loss_d
    
    # Function to validate (calculate FID and generate sample images)
    def validate(epoch):
        netG.eval()
        with torch.no_grad():
            # Generate fake images for visualization
            noise = torch.randn(64, args.latent_dim, 1, 1, device=device)
            fake_images = netG(noise)
            
            # Save sample images
            if not os.path.exists('./samples'):
                os.makedirs('./samples')
            
            torchvision.utils.save_image(fake_images, 
                                        f'./samples/epoch_{epoch+1}.png',
                                        normalize=True, 
                                        nrow=8)
            
            # Calculate FID
            # Get a batch of real images
            real_images_batch = next(iter(train_loader))[0].to(device)
            
            # Generate fake images with the same batch size as real images
            noise = torch.randn(real_images_batch.size(0), args.latent_dim, 1, 1, device=device)
            fake_images_batch = netG(noise)
            
            # Calculate FID score using the FIDCalculator class
            fid_score = fid_calculator(real_images_batch, fake_images_batch, batch_size=32, device=device)
            
            return fid_score
    
    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        loss_g, loss_d = train_one_epoch(epoch)
        
        # Validate and calculate FID
        fid_score = validate(epoch)
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss G: {loss_g:.4f}, Loss D: {loss_d:.4f}, FID: {fid_score:.4f}")
        
        # Save model if it's the best so far (lower FID is better)
        if fid_score < best_fid and args.save_model:
            best_fid = fid_score
            torch.save({
                'epoch': epoch + 1,
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'loss_g': loss_g,
                'loss_d': loss_d,
                'fid_score': fid_score,
                'best_fid': best_fid,
            }, f'./best_gan_model.pth')
            print(f"Best model saved with FID: {best_fid:.4f}")
        
        # Update learning rates
        schedulerG.step()
        schedulerD.step()
        
        # Print current learning rates
        current_lr_g = optimizerG.param_groups[0]['lr']
        current_lr_d = optimizerD.param_groups[0]['lr']
        print(f"Updated learning rates - Generator: {current_lr_g:.6f}, Discriminator: {current_lr_d:.6f}")
        
        # Save model at intervals
        if (epoch + 1) % args.save_interval == 0 and args.save_model:
            torch.save({
                'epoch': epoch + 1,
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'schedulerG': schedulerG.state_dict(),
                'schedulerD': schedulerD.state_dict(),
                'loss_g': loss_g,
                'loss_d': loss_d,
                'fid_score': fid_score,
            }, f'./gan_model_epoch_{epoch+1}.pth')
    
    print("Training completed!")

if __name__ == '__main__':
    freeze_support()
    main()


