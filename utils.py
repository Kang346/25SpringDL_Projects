import matplotlib.pyplot as plt

def plot_loss_accuracy(losses, accuracies):
    epochs = range(1, len(losses) + 1)
    
    # draw loss
    plt.figure()
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.show()
    
    # draw accuracy
    plt.figure()
    plt.plot(epochs, accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy')
    plt.show()