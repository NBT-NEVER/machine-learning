# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：pytoch_test.py
# 开发时间：2025/12/3 17:58
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.backends.cudnn.enabled)

