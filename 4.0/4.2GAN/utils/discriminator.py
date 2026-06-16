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

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        
        self.ndf = ndf
        
        # Main discriminator network with attention
        self.main = nn.Sequential(
            # Block 1: 32x32 -> 32x32
            nn.Conv2d(3, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Add self-attention mechanism
            SelfAttention(ndf * 2),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Add self-attention mechanism
            SelfAttention(ndf * 4),
            
            # Block 4: 8x8 -> 4x4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 5: 4x4 -> 1x1
            nn.Conv2d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final classification layer
            nn.Conv2d(ndf * 16, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
