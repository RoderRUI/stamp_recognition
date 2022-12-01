import paddle
import numpy as np
from matplotlib import pyplot as plt
from paddle.vision.transforms import Normalize

transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
# 下载数据集并初始化 DataSet
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
# 打印数据集里图片数量
print('{} images in train_dataset, {} images in test_dataset'.format(len(train_dataset), len(test_dataset)))
# 模型组网并初始化网络
lenet = paddle.vision.models.LeNet(num_classes=10)
# 可视化模型组网结构和参数
# paddle.summary(lenet, (1, 1, 28, 28))

# 封装模型，将网络结构组合成可快速使用 飞桨高层 API 进行训练、评估、推理的实例，方便后续操作
model = paddle.Model(lenet)

# 模型训练的配置准备，准备损失函数，优化器和评价指标
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# print(img)
# print(label[0])
# 配置循环参数并启动模型训练 batch_size:训练的批大小 epochs: 训练轮数
model.fit(train_dataset, epochs=1, batch_size=1, verbose=1)
# 模型评估
model.evaluate(test_dataset, batch_size=64, verbose=1)

# 保存模型
model.save('./output/mnist')
# 加载模型
model.load('output/mnist')

# 从测试集中取出一张图片
img, label = test_dataset[0]
# 将图片shape从1*28*28变为1*1*28*28，增加一个batch维度，以匹配模型输入格式要求
img_batch = np.expand_dims(img.astype('float32'), axis=0)
print(img_batch)

# 执行推理并打印结果，此处predict_batch返回的是一个list，取出其中数据获得预测结果
out = model.predict_batch(img_batch)[0]
pred_label = out.argmax()
print('true label: {}, pred label: {}'.format(label[0], pred_label))
# 可视化图片
plt.imshow(img[0])

# if __name__ == '__main__':
#     print(1)
