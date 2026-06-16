import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Self-attention mechanism for GANs"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Query, key, value projections
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Scale factor
        self.scale = (in_channels // 8) ** -0.5
        
        # Learnable parameter gamma
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Reshape spatial dimensions into sequence length
        # (batch_size, channels, height * width)
        x_flat = x.view(batch_size, channels, height * width)
        
        # Compute query, key, value
        query = self.query_conv(x).view(batch_size, channels // 8, height * width)
        key = self.key_conv(x).view(batch_size, channels // 8, height * width)
        value = self.value_conv(x).view(batch_size, channels, height * width)
        
        # Compute attention scores
        # (batch_size, height * width, height * width)
        attention = torch.bmm(query.transpose(1, 2), key) * self.scale
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        # (batch_size, channels, height * width)
        out = torch.bmm(value, attention.transpose(1, 2))
        
        # Reshape back to original size
        out = out.view(batch_size, channels, height, width)
        
        # Add residual connection
        out = self.gamma * out + x
        
        return out

class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64):
        super(Generator, self).__init__()
        
        self.ngf = ngf
        self.latent_dim = latent_dim
        
        # Initial dense layer to expand latent space
        self.fc = nn.Linear(latent_dim, ngf * 8 * 4 * 4)
        
        # Main generator network with attention
        self.main = nn.Sequential(
            # Reshape to 4x4 feature map
            nn.Unflatten(1, (ngf * 8, 4, 4)),
            
            # Block 1: 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # Block 2: 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # Add self-attention mechanism
            SelfAttention(ngf * 2),
            
            # Block 3: 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # Add self-attention mechanism
            SelfAttention(ngf),
            
            # Block 4: 32x32 -> 32x32 (final output)
            nn.Conv2d(ngf, 3, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size: 3 x 32 x 32
        )

    def forward(self, input):
        # Expand latent space
        x = self.fc(input.view(input.size(0), -1))
        # Pass through main network
        return self.main(x)
