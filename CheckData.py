from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])
    # 使用ImageFolder函数，读取数据文件夹构建数据集dataset这个函数会将保存数据的文件夹的名字，作为数据的标签，组织数据例如，对于名字为“3
    # "的文件夹
    # 会将“3
    # "就会作为文件夹中图像数据的标签，和图像配对，用于后续的训练，使用起来非常的方便
    train_dataset =datasets.ImageFolder(root='E:\WorkPlace\PythonProject\MyNet\mnist_images\mnist_train',transform=transform)
    test_dataset =datasets.ImageFolder(root='E:\WorkPlace\PythonProject\MyNet\mnist_images\mnist_test',transform=transform)
    print(len(train_dataset))
    print(len(test_dataset))

    #加载数据
    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    print(len(train_loader))
    for batch_idx,(data, label) in enumerate (train_loader):
        if batch_idx==3:
            break
        print("batch_idx",batch_idx)
        print("data.shape",data.shape)
        print("label:",label.shape)
        print(label)
