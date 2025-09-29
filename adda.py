import copy
from pprint import pprint
from tqdm import tqdm
from itertools import cycle

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_adapt.hooks import ADDAHook
from pytorch_adapt.models import Discriminator

from ddc import train, test, G_class as G, C_class as C

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# data
transform_mnist = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

transform_usps = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)

mnist = datasets.MNIST("data", train=True, download=True, transform=transform_mnist)
usps = datasets.USPS("data", train=False, download=True, transform=transform_usps)

mnist_loader = DataLoader(mnist, batch_size=32, shuffle=True, drop_last=True)
usps_loader = DataLoader(usps, batch_size=32, shuffle=True, drop_last=True)

# models
adda_G = G().to(device)
adda_C = C().to(device)
adda_D = Discriminator(in_size=500, h=128).to(device)

G_opt = torch.optim.Adam(adda_G.parameters(), lr=1e-4, betas=(0.5, 0.9))
C_opt = torch.optim.Adam(adda_C.parameters(), lr=1e-4, betas=(0.5, 0.9))
D_opt = torch.optim.Adam(adda_D.parameters(), lr=1e-4, betas=(0.5, 0.9))

print("--- Training model on source data ---")
# training feature extractor and classifier
# need to optimize both model's parameters
optimizer = torch.optim.Adam(list(adda_G.parameters()) + list(adda_C.parameters()), lr = 0.0001)
criterion = nn.CrossEntropyLoss()

train(adda_G, adda_C, mnist_loader, optimizer, criterion, device)

print("--- Testing pretrained model on target data ---")
test(adda_G, adda_C, device, usps_loader)

print("--- Performing Domain Adaptation with ADDA ---")

# target model
# begins as a copy of source's feature extractor
T = copy.deepcopy(adda_G).to(device)
T_opt = torch.optim.Adam(T.parameters(), lr=1e-4, betas=(0.5, 0.9))

hook = ADDAHook(g_opts=[T_opt], d_opts=[D_opt])

num_epochs = 100
for epoch in range(num_epochs):
    # necessary because datasets have different lengths
    # avoid stopping when run out of data
    usps_iter = iter(cycle(usps_loader))
    mnist_iter = iter(mnist_loader)

    for _ in tqdm(range(len(mnist_loader))):
        src_imgs, src_labels = next(mnist_iter)
        target_imgs, _ = next(usps_iter)

        src_imgs = src_imgs.to(device)
        src_labels = src_labels.to(device)
        target_imgs = target_imgs.to(device)

        batch = {
                "src_imgs": src_imgs,
                "src_labels": src_labels,
                "target_imgs": target_imgs,
                # the discriminator must predict which domain a feature came from
                # src domain features will be represented by 0 (label 0)
                # target domain features will be represented by 1 (label 1)
                "src_domain": torch.zeros(
                    src_imgs.size(0), dtype=torch.long, device=device
                ),
                "target_domain": torch.ones(
                    target_imgs.size(0), dtype=torch.long, device=device
                ),
            }

        models = {"G": adda_G, "C": adda_C, "D": adda_D, "T": T}

        _, losses = hook({**models, **batch})

    # epoch loss
    print(f"Epoch {epoch+1}/{num_epochs}:")
    pprint(losses)

    # target data for testing
    test(T, adda_C, device, usps_loader) 
