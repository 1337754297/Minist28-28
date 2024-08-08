import numpy as np
import struct
from PIL import Image
import os

print("Processing started")

# 设置数据文件路径
data_file = r'E:\WorkPlace\PythonProject\MyNet\data\MNIST\raw\train-images-idx3-ubyte'
# 原始数据文件大小是 47040016 字节，但我们要去掉前 16 字节的头部信息
data_file_size = 47040016
data_file_size = str(data_file_size - 16) + 'B'  # 减去头部信息后，得到实际图像数据的大小
data_buf = open(data_file, 'rb').read()  # 以二进制形式读取数据文件
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', data_buf, 0)  # 解析头部信息
datas = struct.unpack_from('>' + data_file_size, data_buf, struct.calcsize('>IIII'))  # 读取图像数据
datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)  # 将数据转换为numpy数组并调整形状

# 设置标签文件路径
label_file = r'E:\WorkPlace\PythonProject\MyNet\data\MNIST\raw\train-labels-idx1-ubyte'
# 原始标签文件大小是 60008 字节，但我们要去掉前 8 字节的头部信息
label_file_size = 60008
label_file_size = str(label_file_size - 8) + 'B'  # 减去头部信息后，得到实际标签数据的大小
label_buf = open(label_file, 'rb').read()  # 以二进制形式读取标签文件
magic, numLabels = struct.unpack_from('>II', label_buf, 0)  # 解析头部信息
labels = struct.unpack_from('>' + label_file_size, label_buf, struct.calcsize('>II'))  # 读取标签数据
labels = np.array(labels).astype(np.int64)  # 将标签数据转换为numpy数组并调整数据类型

# 创建用于保存图像数据的根目录
datas_root = 'mnist_train'
if not os.path.exists(datas_root):
    os.mkdir(datas_root)

# 为每个标签创建一个文件夹，用于保存对应的图像
for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

print("Processing images...")

# 遍历每一个标签，并将对应的图像保存为 PNG 文件
for ii in range(numLabels):
    if ii % 1000 == 0:
        print(f"Processed {ii}/{numLabels} images...")  # 每处理 1000 张图像，输出处理进度
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])  # 将 numpy 数组转换为图像
    label = labels[ii]  # 获取图像对应的标签
    file_name = datas_root + os.sep + str(label) + os.sep + 'mnist_train_' + str(ii) + '.png'  # 构造保存文件名
    img.save(file_name)  # 保存图像

print("Processing completed")  # 输出处理完成信息
