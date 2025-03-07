import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

def load_data(batch_size = 64):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # download CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # dataloader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

def show_image(img_tensor, label):
    # CIFAR-10 类别标签
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 将张量转换为 NumPy 数组并调整维度
    img = img_tensor.permute(1, 2, 0).numpy()
    
    # 反归一化
    img = img * 0.5 + 0.5
    
    # 显示图像
    plt.imshow(img)
    plt.title(f"Label: {classes[label]}")
    plt.axis('off')  # 隐藏坐标轴
    plt.show()

if __name__ == "__main__":

    trainloader, testloader = load_data()
    print(f"Number of training batches: {len(trainloader)}")
    print(f"Number of testing batches: {len(testloader)}")
    print(f"Number of training samples: {len(trainloader.dataset)}")
    print(f"Number of testing samples: {len(testloader.dataset)}")

    # 获取第一张图像及其标签
    first_image, first_label = testloader.dataset[0]
    print(f"Shape of the first training sample: {first_image.shape}")
    print(f"Label of the first training sample: {first_label}")

    # 显示第一张图像及其标签
    show_image(first_image, first_label)