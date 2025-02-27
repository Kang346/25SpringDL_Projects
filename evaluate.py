import torch
from model import ResNetCIFAR10
from dataset import load_data

def evaluate_model():
    # load data
    _, testloader = load_data()
    
    # load model
    model = ResNetCIFAR10()
    model.load_state_dict(torch.load('resnet_cifar.pth'))  # laod the trained model parameters
    model.eval()

    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    evaluate_model()
