# coding: utf-8
import sys, os
import cupy as np
from PIL import Image
import argparse

# 为了导入父目录的文件而进行的设定
sys.path.append(os.pardir)

from dataset.mnist import load_mnist

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MNIST分类器验证脚本')
    parser.add_argument('--network', type=str, default='deep', choices=['simple', 'deep'],
                        help='选择使用的网络类型: simple 或 deep (默认: simple)')
    args = parser.parse_args()
    
    network_type = args.network
    total_images = 10000
    
    # 创建结果文件夹
    result_dir = f'result_{network_type}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 读取MNIST测试集
    print("正在加载MNIST测试集...")
    (_, _), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=False)
    print(f"测试集加载完成：{x_test.shape[0]} 张图像")
    
    # 根据选择的网络类型初始化网络
    print(f"正在初始化 {network_type} 网络并加载参数...")
    
    if network_type == 'simple':
        from simple_convnet import SimpleConvNet
        network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param={'filter_num1':16, 'filter_num2':32, 'filter_num3':32, 
                            'filter_size1':3, 'filter_size2':3, 'filter_size3':3,
                            'pad1':1, 'pad2':1, 'pad3':1, 
                            'stride1':1, 'stride2':1, 'stride3':1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
        param_file = "params.pkl"
    else:  # deep
        from deep_convnet import DeepConvNet
        network = DeepConvNet()
        param_file = "deep_convnet_params.pkl"
    
    try:
        network.load_params(param_file)
        print(f"参数文件 {param_file} 加载成功")
    except Exception as e:
        print(f"参数文件加载失败: {e}")
        sys.exit(1)
    
    # 限制测试图像数量
    total_images = min(total_images, x_test.shape[0])
    print(f"将测试 {total_images} 张图像")
    
    # 测试网络
    error_count = 0
    print("开始测试...")
    for i in range(total_images):
        # 显示进度
        if i % 100 == 0:
            print(f"处理进度: {i}/{total_images}")
        
        # 预测
        img = x_test[i:i+1]
        label = t_test[i]
        pred = network.predict(img)
        pred_label = np.argmax(pred)
        
        # 保存错误分类的图像
        if pred_label != label:
            error_count += 1
            # 将图像转换为可保存的格式
            img = img.reshape(28, 28) * 255  # 反归一化
            img = Image.fromarray(np.uint8(img.get()))
            
            # 生成文件名：ori-本身应该的数字-预测结果-00001.png
            filename = f"ori-{label}-{pred_label}-{error_count:05d}.png"
            filepath = os.path.join(result_dir, filename)
            
            img.save(filepath)
    
    # 计算准确率
    accuracy = (total_images - error_count) / total_images * 100
    print(f"准确率: {accuracy:.2f}%")
    print(f"错误分类的图像数量: {error_count}")
    print(f"{network_type} 网络测试完成！")
    print(f"错误分类图像已保存到 {result_dir} 文件夹")

if __name__ == '__main__':
    main()