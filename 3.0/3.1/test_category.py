# _*_coding:UTF-8_*_
# 开发人员：NBT
# 文件名称：test_category.py
# 开发时间：2025/12/4 19:44
import os
import cv2
import torch
from torchvision import transforms
from dataset import SaliencyDataset
from metric import calc_cc_score, KLD as CC, KLD
# from metric import CC, KLD
# from model import SaliencyResNet18
from model import ResNet18Saliency as SaliencyResNet18
import csv


def test(model_path, test_root, save_root):

    # -----------------------
    # 1. 准备模型
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SaliencyResNet18().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # -----------------------
    # 2. 加载测试集
    # -----------------------
    test_dataset = SaliencyDataset(test_root, img_size=(256, 256), is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )

    # -----------------------
    # 3. 按类别统计的容器
    # -----------------------
    category_metrics = {}
    # 格式:
    # {
    #   "cat": {"CC": [...], "KL": [...]},
    #   "dog": {"CC": [...], "KL": [...]},
    # }

    os.makedirs(save_root, exist_ok=True)

    # -----------------------
    # 4. 测试循环
    # -----------------------
    with torch.no_grad():
        for i, (img, mask, ori_size, mask_ori, img_ori) in enumerate(test_loader):

            # 获取原图路径
            img_path = test_dataset.img_paths[i]

            # 类别名（Stimuli 下的子文件夹名）
            category_name = os.path.basename(os.path.dirname(img_path))

            # 图像文件名
            img_name = os.path.basename(img_path)

            # 原始尺寸
            ori_h, ori_w = int(ori_size[0]), int(ori_size[1])

            # 数据迁移
            img = img.to(device)
            mask_gt = mask_ori.squeeze().numpy()

            # 前向推理
            pred = model(img).squeeze().cpu().numpy()

            # 恢复到原尺寸
            pred_resized = cv2.resize(pred, (ori_w, ori_h))

            # 计算指标
            cc_val = calc_cc_score(mask_gt, pred_resized)
            kl_val = KLD(mask_gt, pred_resized)

            # 按类别保存
            if category_name not in category_metrics:
                category_metrics[category_name] = {"CC": [], "KL": []}

            category_metrics[category_name]["CC"].append(cc_val)
            category_metrics[category_name]["KL"].append(kl_val)

            # 保存预测显著图
            save_dir = os.path.join(save_root, category_name)
            os.makedirs(save_dir, exist_ok=True)

            pred_img = (pred_resized * 255).astype("uint8")
            cv2.imwrite(os.path.join(save_dir, img_name), pred_img)

    # -----------------------
    # -----------------------
    # 5. 输出最终统计结果（CSV 表格）
    # -----------------------
    result_csv = os.path.join(save_root, "metrics_by_category.csv")

    with open(result_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # 表头
        writer.writerow(["Category", "Average_CC", "Average_KL"])

        for cls in category_metrics:
            cc_mean = sum(category_metrics[cls]["CC"]) / len(category_metrics[cls]["CC"])
            kl_mean = sum(category_metrics[cls]["KL"]) / len(category_metrics[cls]["KL"])

            writer.writerow([cls, f"{cc_mean:.4f}", f"{kl_mean:.4f}"])
            print(f"类别：{cls} | 平均CC={cc_mean:.4f} | 平均KL={kl_mean:.4f}")

    print("\n===> 按类别指标统计完成，结果已保存为表格：", result_csv)


if __name__ == "__main__":
    model_path = "resnet18_saliency_best.pth"
    test_root = "3-Saliency-TestSet"
    save_root="./category"
    test(model_path, test_root, save_root)