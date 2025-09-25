import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from ignite.metrics import MaximumMeanDiscrepancy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from adaptiode import  sim2real

from model import *
from torchvision.models import resnet50, ResNet50_Weights

torch.manual_seed(1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## model pretrained on simulation data
model = resnet50()
model.load_state_dict(torch.load('Resnet_pretrained_sim2real.pt'))
model = model.to(device)

## real (target) data
_, target_data = sim2real(batch_size=32, num_workers=12, use_imagenet_norm=True)

test(model, device, target_data)