import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import *

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')   
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
    return parser.parse_args()

args = parser()
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