import torch
from torchvision import transforms
from PIL import Image
from model import Network

def predict(image_path, model_path):
    # 定义与训练时相同的图像预处理变换
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度图像
        transforms.ToTensor()  # 将图像转换为张量
    ])

    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 判断是否使用GPU
    model = Network().to(device)  # 初始化模型并将其加载到设备上
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型参数
    model.eval()  # 将模型设置为评估模式

    # 加载并预处理图像
    image = Image.open(image_path).convert('L')  # 打开图像并转换为灰度图像
    image = transform(image).unsqueeze(0)  # 预处理图像并添加批次维度
    image = image.to(device)  # 将图像加载到设备上

    # 预测
    with torch.no_grad():  # 关闭梯度计算（评估模式下不需要梯度）
        output = model(image)  # 将图像输入模型，得到输出
        _, predicted = torch.max(output.data, 1)  # 获取预测结果的最大值对应的类别
        return predicted.item()  # 返回预测的类别标签

# 使用函数
image_path = r'E:\WorkPlace\PythonProject\MyNet\mnist_images\mnist_test\7\mnist_test_26.png'  # 设置图像路径
model_path = 'mnist.pth'  # 设置模型路径
predicted_label = predict(image_path, model_path)  # 调用预测函数
print(f"The predicted label is: {predicted_label}")  # 输出预测结果
