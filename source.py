import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from adaptiode import *

from model import *
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(model_name):
    ## model
    if model_name == "Alexnet":
        ## data
        train_loader, test_loader, target = sim2real_split(batch_size=32, num_workers=12)
        num_classes = len(target.dataset.classes)    

        model = AlexNet(num_classes=num_classes).to(device)
    elif model_name == "Resnet":
        ## data with imagenet normalization
        train_loader, test_loader, target = sim2real_split(batch_size=32, num_workers=12, use_imagenet_norm=True)

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    else:
        print("No such model name defined")
        exit()

    optimzer = optim.Adam(model.parameters(), lr = 0.0001) 
    criterion = nn.CrossEntropyLoss()

    ## data
    show_dataloader_samples(train_loader)
    
    ## training
    for epoch in range(0, 5):
        train(model, device, train_loader, optimzer, criterion, epoch)
        test(model, device, test_loader)

    ## save model
    file_name = f"{model_name}_pretrained_sim2real.pt"
    torch.save(model.state_dict(), file_name)

if __name__ == "__main__":
    main(model_name="Resnet")