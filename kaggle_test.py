import pickle 
import torch
from model import CustomResNet
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from PIL import Image

# read the kaggle custom dataset
with open('rh/data/cifar_test_nolabel.pkl', 'rb') as f:
    data = pickle.load(f)

images = data[b'data']
ids = data[b'ids']

# print the number of images an the shape of image
print(f'Number of images: {len(images)}')
print(f'Shape of image: {images[0].shape}')

# convert the data to tensor
images = images.reshape(-1, 32, 32, 3).astype(np.float32)
images = images / 255.0  

transform = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# convert the images to tensor
images = np.transpose(images, (0, 3, 1, 2))
images = torch.tensor(images, dtype=torch.float32)

images = torch.stack([transform(img) for img in images])


# Convert image tensor to a displayable format
def show_image(img_tensor):
    # Convert from (C, H, W) to (H, W, C) for displaying with matplotlib
    img = img_tensor.permute(1, 2, 0).numpy()

    # Normalize the image to [0, 1]
    img = (img - img.min()) / (img.max() - img.min())  

    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Hide axes for better image display
    plt.show()

# show the first n image to check
for _ in range(1):
    show_image(images[_])

# load the model
model = CustomResNet()
model.load_state_dict(torch.load('best_model.pth'))

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# predict the labels
model.eval()
predictions = []
with torch.no_grad():
    for inputs in images:
        inputs = inputs.unsqueeze(0).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted.item())

# save the predictions to a csv file
with open('submission.csv', 'w') as f:
    f.write('ID,Labels\n')
    for i, pred in zip(ids, predictions):
        f.write(f'{i},{pred}\n')