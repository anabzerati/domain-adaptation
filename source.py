import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import *

args = parse_args()
torch.manual_seed(args.seed)
 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

train_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('./data', train=True, download=True, transform=transf),
                    batch_size=args.batch_size, shuffle=True)
    
test_loader = torch.utils.data.DataLoader( 
                    datasets.MNIST('./data', train=False, download=True, transform=transf), 
                    batch_size=1, shuffle=False)

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

if (args.save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")   