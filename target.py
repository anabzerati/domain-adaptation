import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from ignite.metrics import MaximumMeanDiscrepancy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import *

args = parser()
torch.manual_seed(args.seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pt'))

model = model.to(device)

transf = transforms.Compose([
                            transforms.Resize((28, 28)),
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])

test_data = datasets.SVHN('data', split='test', download=True, transform=transf)
loader = DataLoader(test_data, batch_size=1, shuffle=False)

for i in range(5):
    image, label = test_data[i]

    # Convert the PyTorch tensor to a NumPy array for visualization
    image_np = image.squeeze().numpy() 

    plt.imshow(image_np, cmap='gray')
    plt.title(f"Label: {label}")
    plt.show()

test(model, device, loader)
