import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary

import config
from dataset import load_data
from model import CustomResNet
from utils import plot_loss_accuracy
from evaluate import evaluate_model

def train_model(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, learning_rate=config.LEARNING_RATE):
    print("Loading data...")
    # load data
    trainloader, testloader = load_data(batch_size)
    print("Data loaded successfully.")
    
    print("Loading model...")
    # load model
    model = CustomResNet()
    criterion = nn.CrossEntropyLoss()  # use cross entropy loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)  # use SGD optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print("Model loaded successfully.")
    
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
    
    # training
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # use tqdm to show progress bar
        for inputs, labels in tqdm(trainloader, desc=f'Epoch [{epoch+1}/{epochs}]', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
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
    # plot loss and accuracy
    plot_loss_accuracy(losses, train_acc)

    print("Finished Training")
    torch.save(model.state_dict(), 'resnet_cifar.pth')  # save the model

if __name__ == "__main__":
    train_model()