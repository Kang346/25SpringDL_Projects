import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
from torch.optim.lr_scheduler import LinearLR
import wandb
import numpy as np
import torch.nn.functional as F
import random

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from utils import cutmix_data
import config
from dataset import load_data
from model import CustomResNet
from evaluate import evaluate_model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, learning_rate=config.LEARNING_RATE):
    # init wandb
    wandb.init(project="your_project_name", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    })

    print("Loading data...")
    # load data
    trainloader, testloader = load_data(batch_size)
    print("Data loaded successfully.")
    
    print("Loading model...")
    # load model
    model = CustomResNet()
    criterion = nn.CrossEntropyLoss()  # use cross entropy loss
    kl_criterion = nn.KLDivLoss(reduction='batchmean')  # use KLDivLoss for soft labels
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)  # use SGD optimizer
    print("Model loaded successfully.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    
    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # print model summary
    summary(model, (3, 32, 32))

    # store loss and accuracy
    losses = []
    train_acc = []
    test_acc = []
    best_acc = 0.0
    

    # load softmax from teacher model
    # teacher_softmax = np.load("teacher_softmax_outputs.npy")  
    # teacher_softmax = torch.tensor(teacher_softmax, dtype=torch.float32)  
    # teacher_softmax = TensorDataset(teacher_softmax)

    # print("teacher_softmax shape:", teacher_softmax.shape)
    # print("trainloader dataset shape:", len(trainloader.dataset))

    # teacher_softmax_loader = torch.utils.data.DataLoader(teacher_softmax, batch_size=batch_size, shuffle=False)


    # temperature = 3.0 

    # training
    for epoch in range(epochs):
        # 设置随机种子
        # seed = epoch
        # set_seed(seed)

        # # 创建统一的索引
        # indices = torch.randperm(len(trainloader.dataset))

        # # 按照统一的索引打乱数据
        # trainloader = DataLoader(trainloader.dataset, batch_size=batch_size, sampler=SubsetRandomSampler(indices))
        # teacher_softmax_loader = DataLoader(teacher_softmax, batch_size=batch_size, sampler=SubsetRandomSampler(indices))


        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # use tqdm to show progress bar
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            if config.CUTMIX_PROB > 0.0 and torch.rand(1).item() < config.CUTMIX_PROB:
                alpha = 1.0
                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha)
                outputs = model(inputs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)


            weights = torch.ones_like(labels, dtype=torch.float32)
            weights[labels == 3] = 3.0  
            # weights[labels == 9] = 3.0  
            weights[labels == 2] = 3.0  
            loss = (loss * weights).mean()

            # zero the parameter gradients
            optimizer.zero_grad()

            # backward
            # loss.backward()

            # 获取当前批次的 teacher_softmax 输出
            # batch_teacher_softmax = batch_teacher_softmax.to(device)


            # assert inputs.size(0) == batch_teacher_softmax.size(0), f"Batch size mismatch: {inputs.size(0)} != {batch_teacher_softmax.size(0)}"

            # 获取学生模型的输出
            # student_output = model(inputs)  # inputs 是当前训练数据

            # 计算学生模型的 softmax 输出
            # student_softmax = F.log_softmax(student_output / temperature, dim=1)  # 使用温度调节


            # 计算 KL 散度损失
            # soft_loss = kl_criterion(student_softmax, batch_teacher_softmax)  # 计算 KL 散度损失

            # 计算硬标签的交叉熵损失
            # hard_loss = F.cross_entropy(student_output, labels)  # hard_labels 是传统的硬标签

            # 合并损失（软标签 + 硬标签）
            # alpha = 0.7  # 软标签的权重
            # final_loss = alpha * soft_loss + (1 - alpha) * hard_loss

            # 反向传播并更新参数 
            loss.backward()

            # optimize
            optimizer.step()

            # statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        
        
        # store loss and accuracy
        losses.append(running_loss/len(trainloader))
        train_acc.append(100 * correct / total)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%")

        wandb.log({
            "epoch": epoch + 1,
            "loss": running_loss / len(trainloader),
            "train_loss": running_loss / len(trainloader),
            "train_accuracy": 100 * correct / total,
        })

        # Step the scheduler
        scheduler.step()

        # Evaluate on test data
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        test_acc.append(test_accuracy)
        print(f"Test Accuracy: {test_accuracy:.2f}%")

        wandb.log({
            "test_accuracy": test_accuracy
        })

        # Save the best model
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'best_model.pth')
            print(f"New best model saved with accuracy {best_acc:.2f}%")
        
        # Early stopping
        # if test_accuracy <= best_acc:
        #     no_improve += 1
        #     if no_improve >= patience:
        #         print(f"Early stopping at epoch {epoch}")
        #         break
        # else:
        #     no_improve = 0

    print("Finished Training")
    torch.save(model.state_dict(), 'resnet_cifar.pth')  # save the model

if __name__ == "__main__":
    train_model()