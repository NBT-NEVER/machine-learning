import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 归一化处理
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. 构建模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 4. 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 6. 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\n测试准确率:', test_acc)

# 7. 预测并展示结果
predictions = model.predict(x_test)

# 显示前10个测试图片及预测结果
for i in range(10):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"真实标签: {y_test[i]}, 预测: {predictions[i].argmax()}")
    plt.show()