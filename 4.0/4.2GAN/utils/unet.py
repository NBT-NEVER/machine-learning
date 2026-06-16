import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """即Transformer中的位置编码，用于将时间步t映射为向量"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """基本的ResNet块"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        # 第一次卷积
        h = self.bnorm1(self.relu(self.conv1(x)))
        # 注入时间信息
        time_emb = self.relu(self.time_mlp(t))
        # 扩展时间emb维度以匹配特征图: (batch, ch) -> (batch, ch, 1, 1)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # 添加时间信息
        h = h + time_emb
        # 第二次卷积
        h = self.bnorm2(self.relu(self.conv2(h)))
        # 上采样或下采样
        return self.transform(h)

class SimpleUNet(nn.Module):
    """简化的U-Net架构"""
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3 
        time_emb_dim = 32

        # 时间嵌入层
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # 初始卷积
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample 路径
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) for i in range(len(down_channels)-1)])
        
        # Upsample 路径
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) for i in range(len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # 时间嵌入
        t = self.time_mlp(timestep)
        
        # 初始卷积
        x = self.conv0(x)
        
        # 保存残差连接
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
            
        # Upsample
        for up in self.ups:
            residual = residuals.pop()
            # 拼接残差 (Skip Connection)
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
            
        return self.output(x)

if __name__ == '__main__':
    # 测试代码
    net = SimpleUNet()
    print("Num params: ", sum(p.numel() for p in net.parameters()))
    print("Num params: ", sum(p.numel() for p in net.parameters()))
    x = torch.randn(3, 3, 32, 32)
    t = torch.randint(0, 1000, (3,))
    print(net(x, t).shape)
