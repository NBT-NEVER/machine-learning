#%% md
# 先定义一个处理数据集的类SaliencyDataset，用这个类读取数据集中的图像，运行成功得到下面的输出：
# ```
# 成功加载1600个样本
# 成功加载400个样本
# ```
#%%
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os


class SaliencyDataset(Dataset):
    def __init__(self, root_dir, img_size=(256, 256), is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train

        # 递归获取所有图像路径
        self.img_paths = []
        img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
        for root, _, files in os.walk(os.path.join(root_dir, "Stimuli")):
            for file in files:
                if file.lower().endswith(img_extensions):
                    self.img_paths.append(os.path.join(root, file))

        # 匹配掩码路径
        self.mask_paths = []
        for img_path in self.img_paths:
            mask_path = img_path.replace("Stimuli", "FIXATIONMAPS")
            mask_path = os.path.splitext(mask_path)[0]
            found = False
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                candidate = mask_path + ext
                if os.path.exists(candidate):
                    self.mask_paths.append(candidate)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"未找到{img_path}对应的掩码文件")

        # 数据增强
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
        ]) if is_train else None

        print(f"成功加载{len(self.img_paths)}个样本")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取原图并记录尺寸
        img_path = self.img_paths[idx]
        img_ori = cv2.imread(img_path)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img_ori.shape[:2]  # 保存原图尺寸

        # 预处理输入图像（resize到256*256）
        img = cv2.resize(img_ori, self.img_size)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 读取并预处理掩码
        mask_path = self.mask_paths[idx]
        mask_ori = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask_ori, self.img_size)
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        # 数据增强
        if self.transform and self.is_train:
            seed = torch.randint(0, 1000000, (1,)).item()
            torch.manual_seed(seed)
            img = self.transform(img)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        # 返回原图尺寸和原始掩码（用于测试指标计算）
        return img, mask, (ori_h, ori_w), mask_ori, img_ori

IMG_SIZE = (256, 256)  # 图像尺寸
BATCH_SIZE = 16  # 批次大小
os.chdir(r"I:\STUDY\python\project\machine-learning\3.0\3.2")
print(os.listdir(os.getcwd()))
print(os.getcwd())
train_dataset = SaliencyDataset(r'../3.1/3-Saliency-TrainSet', img_size=IMG_SIZE, is_train=True)
val_dataset = SaliencyDataset(r'../3.1/3-Saliency-TestSet', img_size=IMG_SIZE, is_train=False)  # 若有独立验证集可替换路径
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
#%% md
# 在此处定义显著性预测网络ResNet18Saliency，可以得到测试的输入尺寸、网络结构、参数量和输出尺寸，用于检查网络本身是否有维度问题：
# ```
# 输入尺寸:
# torch.Size([10, 3, 256, 256])
# 
# 网络结构:
# 略
# 
# 参数量:
# 总参数量: 12,062,273
# 可训练参数量: 12,062,273
# 
# 输出尺寸:
# torch.Size([10, 1, 256, 256])
# 
# 输出结果示例:
# 样本 1示例 = tensor([[[0.3826, 0.5688, 0.3519,  ..., 0.5143, 0.4209, 0.5343],
#          [0.8173, 0.2413, 0.9399,  ..., 0.1048, 0.9848, 0.1829],
#          [0.5116, 0.7266, 0.4527,  ..., 0.2846, 0.1125, 0.4992],
#          ...,
#          [0.5290, 0.8419, 0.5386,  ..., 0.1244, 0.8128, 0.4312],
#          [0.7825, 0.6141, 0.1809,  ..., 0.6196, 0.2148, 0.7859],
#          [0.2452, 0.4835, 0.5813,  ..., 0.3819, 0.9410, 0.3715]]],
#        device='cuda:0', grad_fn=<SelectBackward0>)
# ```
#%%
import torch
import torch.nn as nn
import torchvision.models as models
class ResNet18Saliency(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练ResNet18并拆分编码器
        resnet = models.resnet18(pretrained=pretrained)
        resnet.load_state_dict(torch.load('/kaggle/input/resnet18-f37072fd-class/pytorch/default/1/resnet18-f37072fd.pth', map_location=device))
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64通道, 1/2
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64通道, 1/4
        self.encoder3 = resnet.layer2  # 128通道, 1/8
        self.encoder4 = resnet.layer3  # 256通道, 1/16
        self.encoder5 = resnet.layer4  # 512通道, 1/32

        # 解码器：上采样+特征融合
        self.decoder5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.ConvTranspose2d(256 + 256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(64 + 64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.ConvTranspose2d(64 + 64, 1, kernel_size=2, stride=2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器提取多尺度特征
        feat1 = self.encoder1(x)
        feat2 = self.encoder2(feat1)
        feat3 = self.encoder3(feat2)
        feat4 = self.encoder4(feat3)
        feat5 = self.encoder5(feat4)

        # 解码器融合与上采样
        dec5 = self.decoder5(feat5)
        fuse4 = torch.cat([dec5, feat4], dim=1)
        dec4 = self.decoder4(fuse4)

        fuse3 = torch.cat([dec4, feat3], dim=1)
        dec3 = self.decoder3(fuse3)

        fuse2 = torch.cat([dec3, feat2], dim=1)
        dec2 = self.decoder2(fuse2)

        fuse1 = torch.cat([dec2, feat1], dim=1)
        out = self.decoder1(fuse1)

        return self.sigmoid(out)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.rand(10, 3, 256, 256).to(device)
model = ResNet18Saliency(pretrained=False).to(device)

print("输入尺寸:")
print(f"{dummy_input.shape}")
print()

# 打印网络结构
print("网络结构:")
print(model)

# 计算并打印参数量
total_params, trainable_params = count_parameters(model)
print("参数量:")
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print()

# 前向传播
output = model(dummy_input)

# 打印输出尺寸和结果
print("输出尺寸:")
print(f"{output.shape}")
print()
print("输出结果示例:")
print(f"样本 {0+1}示例 = {output[0]}")

#%% md
# 此处为必要的度量指标，没有输出，但是需要运行
#%%
import math
import numpy as np


def calc_cc_score(gtsAnn, resAnn):
    # gtsAnn: Ground-truth saliency map
    # resAnn: Predicted saliency map

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]


EPSILON = np.finfo('float').eps

def KLD(p, q):
    # q: Predicted saliency map
    # p: Ground-truth saliency map
    p = normalize(p, method='sum')
    q = normalize(q, method='sum')
    return np.sum(np.where(p != 0, p * np.log((p+EPSILON) / (q+EPSILON)), 0))

def normalize(x, method='standard', axis=None):

    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res
#%% md
# 训练过程，通过不断进行train_one_epoch进行优化，并使用validate进行验证：
# ```
# Epoch 1/10
# training: 100%|██████████| 100/100 [00:22<00:00,  4.48it/s, batch_loss=0.00616]
# validation: 100%|██████████| 25/25 [00:05<00:00,  4.49it/s]
# 训练损失: 0.0130 | 验证损失: 0.0060
# 保存最佳模型（验证损失：0.0060）
# ```
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="training")
    for imgs, masks, _, _, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"batch_loss": loss.item()})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="validation")
    for imgs, masks, _, _, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
    return total_loss / len(dataloader)


# ===================== 主函数 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "resnet18_saliency_best.pth"  # 最佳模型保存路径


    
EPOCHS = 10  # 训练轮数
LR = 1e-3  # 学习率
criterion = nn.MSELoss() # 损失函数
train_losses = []
val_losses = []

optimizer = optim.Adam(model.parameters(), lr=LR) # 优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率衰减

# 训练主循环
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    print(f"\n Epoch {epoch + 1}/{EPOCHS}")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, SAVE_PATH)
        print(f"保存最佳模型（验证损失：{best_val_loss:.4f}）")
#%% md
# 打印训练过程中训练和验证loss曲线
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training/Validation Loss')
plt.legend()
plt.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)
plt.show()
#%% md
# 使用训练好的网络进行测试，并保存预测结果。
#%%
import torch
import cv2
import numpy as np
import os
from tqdm import tqdm

@torch.no_grad()
def test_and_evaluate(model, test_dir, save_dir="./saliency_results", img_size=(256, 256)):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # 加载测试集（获取原图尺寸和原始掩码）
    test_dataset = SaliencyDataset(test_dir, img_size=img_size, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 初始化指标存储
    all_cc = []
    all_kl = []

    pbar = tqdm(test_loader, desc="测试与评估")
    for idx, (img, _, (ori_h, ori_w), mask_ori, img_ori) in enumerate(pbar):
        category = os.path.basename(os.path.dirname(test_dataset.img_paths[idx]))
        cate_save_dir = os.path.join(save_dir, category)
        os.makedirs(cate_save_dir, exist_ok=True)
        # 模型预测（256*256）
        img = img.to(device)
        saliency_pred = model(img).squeeze().cpu().numpy()  # [256, 256]

        # 将预测结果resize回原图尺寸
        saliency_pred_ori = cv2.resize(saliency_pred, (ori_w.item(), ori_h.item()))  # [ori_h, ori_w]
        mask_ori = mask_ori.squeeze().cpu().numpy()


        # 计算指标
        cc_score = calc_cc_score(mask_ori, saliency_pred_ori)
        kl_score = KLD(mask_ori, saliency_pred_ori)
        all_cc.append(cc_score)
        all_kl.append(kl_score)

        # 获取原始图像文件名和扩展名
        original_file_path = test_dataset.img_paths[idx]
        original_filename = os.path.basename(original_file_path)
        img_name, img_ext = os.path.splitext(original_filename)
        
        # 保存结果
        img_name = os.path.splitext(os.path.basename(test_dataset.img_paths[idx]))[0]

        # 保存resize回原尺寸的显著性图
        saliency_pred_save = (saliency_pred_ori * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(cate_save_dir, f"{img_name}.png"), saliency_pred_save)


    # 计算平均指标
    avg_cc = np.mean(all_cc)
    avg_kl = np.mean(all_kl)
    print(f"\n测试集平均CC系数：{avg_cc:.4f}")
    print(f"测试集平均KL散度：{avg_kl:.4f}")

    # 保存指标结果
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"平均CC系数：{avg_cc:.4f}\n")
        f.write(f"平均KL散度：{avg_kl:.4f}\n")
        f.write(f"所有CC值：{all_cc}\n")
        f.write(f"所有KL值：{all_kl}\n")

    return avg_cc, avg_kl


# ===================== 主函数 =====================

save_path = "/kaggle/working/resnet18_saliency_best.pth"  # 最佳模型保存路径

# 加载最佳模型并测试
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"\n加载最佳模型（Epoch {checkpoint['epoch']+1}）")
avg_cc, avg_kl = test_and_evaluate(model, '/kaggle/input/saliency-class/3-Saliency-TestSet/3-Saliency-TestSet', save_dir="./saliency_results")
print("测试完成！结果已保存至 saliency_results 目录")