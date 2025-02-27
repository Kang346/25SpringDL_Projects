import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import load_data
from model import ResNetCIFAR10
from utils import plot_loss_accuracy

def train_model(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, learning_rate=config.LEARNING_RATE):
    print("Loading data...")
    # load data
    trainloader, testloader = load_data(batch_size)
    print("Data loaded successfully.")
    
    print("Loading model...")
    # load model
    model = ResNetCIFAR10()
    criterion = nn.CrossEntropyLoss()  # use cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # use Adam optimizer
    print("Model loaded successfully.")
    
    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # store loss and accuracy
    losses = []
    accuracies = []
    
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
        accuracies.append(100 * correct / total)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    # plot loss and accuracy
    plot_loss_accuracy(losses, accuracies)

    print("Finished Training")
    torch.save(model.state_dict(), 'resnet_cifar.pth')  # save the model

if __name__ == "__main__":
    train_model()