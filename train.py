import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from model import Network

if __name__ == '__main__':
    # 定义数据的预处理操作。这里，图像首先被转换为灰度图像，然后转换为张量。
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # 加载训练数据集。ImageFolder 函数会自动根据子文件夹的名字给图像打上标签（例如 "3/" 文件夹中的图像会被打上标签 3）。
    train_dataset = datasets.ImageFolder(root='E:\WorkPlace\PythonProject\MyNet\mnist_images\mnist_train',
                                         transform=transform)

    # 可选：如果您还想加载测试数据集，可以取消以下代码的注释。
    # test_dataset = datasets.ImageFolder(root='E:\WorkPlace\PythonProject\MyNet\mnist_images\mnist_test', transform=transform)

    # 使用 DataLoader 加载数据集，每次提取64张图像，并随机打乱数据。
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 判断是否可以使用 GPU，如果可以，则使用 GPU，否则使用 CPU。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化神经网络模型、优化器和损失函数。
    model = Network()  # 加载定义好的神经网络模型
    optimizer = optim.Adam(model.parameters())  # 使用 Adam 优化器
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    model.to(device)  # 将模型加载到设备（GPU 或 CPU）

    # 训练模型，共进行 10 个 epoch。
    for epoch in range(10):
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)  # 将数据和标签加载到设备
            output = model(data)  # 将数据输入模型，得到输出
            loss = criterion(output, label)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            optimizer.zero_grad()  # 清除优化器中的梯度

            # 每 100 个批次打印一次当前的训练状态。
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/10"
                      f" Batch {batch_idx}/{len(train_loader)}"
                      f" Loss: {loss.item():.4f}")

    # 保存训练好的模型参数到文件 "mnist.pth" 中。
    torch.save(model.state_dict(), "mnist.pth")