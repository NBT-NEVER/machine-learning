# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import pickle
import cupy as np
from collections import OrderedDict
from common.layers import *

from common.layers import BatchNormalization

class DeepConvNet:
    """识别率为99%以上的高精度的ConvNet

    网络结构如下所示
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=50, output_size=10):
        # 初始化权重===========
        # 各层的神经元平均与前一层的几个神经元有连接（TODO:自动计算）
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # 使用ReLU的情况下推荐的初始值
        
        self.params = {}
        pre_channel_num = input_dim[0]
        # ====== 添加 BatchNorm 参数（提前初始化） ======
        self.params['gamma1'] = np.ones(conv_param_1['filter_num'])
        self.params['beta1'] = np.zeros(conv_param_1['filter_num'])

        self.params['gamma2'] = np.ones(conv_param_2['filter_num'])
        self.params['beta2'] = np.zeros(conv_param_2['filter_num'])

        self.params['gamma3'] = np.ones(conv_param_3['filter_num'])
        self.params['beta3'] = np.zeros(conv_param_3['filter_num'])

        self.params['gamma4'] = np.ones(conv_param_4['filter_num'])
        self.params['beta4'] = np.zeros(conv_param_4['filter_num'])

        self.params['gamma5'] = np.ones(conv_param_5['filter_num'])
        self.params['beta5'] = np.zeros(conv_param_5['filter_num'])

        self.params['gamma6'] = np.ones(conv_param_6['filter_num'])
        self.params['beta6'] = np.zeros(conv_param_6['filter_num'])

        # ====== 然后才开始循环初始化卷积参数 ======
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([
            conv_param_1, conv_param_2, conv_param_3,
            conv_param_4, conv_param_5, conv_param_6
        ]):
            self.params['W' + str(idx + 1)] = wight_init_scales[idx] * np.random.randn(
                conv_param['filter_num'], pre_channel_num,
                conv_param['filter_size'], conv_param['filter_size']
            )
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        # ====== 添加全连接层参数 ======
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64 * 4 * 4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 生成层===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(BatchNormalization(self.params['gamma1'], self.params['beta1']))
        self.layers.append(Relu())

        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(BatchNormalization(self.params['gamma2'], self.params['beta2']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(BatchNormalization(self.params['gamma3'], self.params['beta3']))
        self.layers.append(Relu())

        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(BatchNormalization(self.params['gamma4'], self.params['beta4']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(BatchNormalization(self.params['gamma5'], self.params['beta5']))
        self.layers.append(Relu())

        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(BatchNormalization(self.params['gamma6'], self.params['beta6']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.55))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.55))
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 自动提取参数层
        grads = {}
        idx = 1
        for layer in self.layers:
            if hasattr(layer, 'dW'):
                grads['W' + str(idx)] = layer.dW
                grads['b' + str(idx)] = layer.db
                idx += 1
        # 自动提取 BatchNorm 层的参数梯度
        bn_idx = 1
        for layer in self.layers:
            if isinstance(layer, BatchNormalization):
                grads['gamma' + str(bn_idx)] = layer.dgamma
                grads['beta' + str(bn_idx)] = layer.dbeta
                bn_idx += 1

        return grads

    def save_params(self, file_name):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BatchNormalization):
                params['running_mean' + str(i + 1)] = layer.running_mean
                params['running_var' + str(i + 1)] = layer.running_var
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    # def load_params(self, file_name="params.pkl"):
    #     with open(file_name, 'rb') as f:
    #         params = pickle.load(f)
    #     for key, val in params.items():
    #         self.params[key] = val
    #
    #     for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
    #         self.layers[layer_idx].W = self.params['W' + str(i+1)]
    #         self.layers[layer_idx].b = self.params['b' + str(i+1)]
    #     # === 新增 === #
    #     for i, layer in enumerate(self.layers):
    #         if isinstance(layer, BatchNormalization):
    #             layer.running_mean = self.params.get('running_mean' + str(i + 1))
    #             layer.running_var = self.params.get('running_var' + str(i + 1))
    def load_params(self, file_name="params.pkl"):
        import pickle

        # === 读取参数文件 ===
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        # === 写回 self.params ===
        for key, val in params.items():
            self.params[key] = val

        # === 按照当前网络层的实际顺序写回权重 ===
        # 你现在的网络层顺序为：
        # 0:Conv1, 1:BN1, 2:Relu, 3:Conv2, 4:BN2, 5:Relu, 6:Pool,
        # 7:Conv3, 8:BN3, 9:Relu, 10:Conv4, 11:BN4, 12:Relu, 13:Pool,
        # 14:Conv5, 15:BN5, 16:Relu, 17:Conv6, 18:BN6, 19:Relu, 20:Pool,
        # 21:Affine1, 22:Relu, 23:Dropout, 24:Affine2, 25:Dropout
        #
        # 因此带权重层索引依次为：
        layer_idxs = (0, 3, 7, 10, 14, 17, 21, 24)

        # === 把保存的权重写回这些层 ===
        for i, layer_idx in enumerate(layer_idxs):
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            if key_w in self.params and key_b in self.params:
                self.layers[layer_idx].W = self.params[key_w]
                self.layers[layer_idx].b = self.params[key_b]
            else:
                print(f"[警告] 参数文件缺少 {key_w} 或 {key_b} ，跳过该层赋值。")

        # === 恢复 BatchNormalization 的运行统计信息 ===
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BatchNormalization):
                mean_key = f'running_mean{i+1}'
                var_key = f'running_var{i+1}'
                if mean_key in self.params:
                    layer.running_mean = self.params[mean_key]
                if var_key in self.params:
                    layer.running_var = self.params[var_key]

        # === 加载完成信息 ===
        print("参数文件加载成功！模型权重与 BatchNorm 状态已恢复。")
