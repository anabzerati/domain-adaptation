import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from adaptiode import *

from model import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():

    train_loader, test_loader, target = sim2real_split(batch_size=32, num_workers=12)


    num_classes = len(target.dataset.classes)    

    model = AlexNet(num_classes=num_classes).to(device)

    optimzer = optim.Adam(model.parameters(), lr = 0.0001) 


    
    show_dataloader_samples(train_loader)
    
    for epoch in range(0, 100):
        train(model, device, train_loader, optimzer, epoch)
        test(model, device, test_loader)
    


if __name__ == "__main__":
    main()