from itertools import cycle
from pprint import pprint
import os
from tqdm import tqdm

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pytorch_adapt.hooks import DANNHook
from pytorch_adapt.models import Discriminator

from ddc import test, train, G_class as G, C_class as C


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
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

    mnist_loader = DataLoader(mnist, batch_size=32, shuffle=True)
    usps_loader = DataLoader(usps, batch_size=32, shuffle=True)

    # The network must be divided on a feature extractor G and a classification head G
    dann_G = G().to(device)
    dann_C = C().to(device)

    # The domain adaption component from the original implementation
    # consists of  three (x→500→1024→1) fully connected layer
    dann_D = Discriminator(in_size=500, h=1024).to(device)

    save_path = "mnist_split.pth"
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)

        dann_G.feature_extractor.load_state_dict(checkpoint["G"])
        dann_C.load_state_dict(checkpoint["C"])

        print("--- Testing pretrained model on MNIST ---")

        test(dann_G, dann_C, device, mnist_loader)
        
    else:
        print("--- Training model on source data ---")
        # training feature extractor and classifier
        # need to optimize both model's parameters
        optimizer = torch.optim.Adam(list(dann_G.parameters()) + list(dann_C.parameters()), lr = 0.0001)
        criterion = nn.CrossEntropyLoss()

        train(dann_G, dann_C, mnist_loader, optimizer, criterion, device)

    print("--- Testing USPS before domain adaptation ---")
    test(dann_G, dann_C, device, usps_loader)
    
    G_opt = torch.optim.Adam(dann_G.parameters(), lr=1e-3)
    C_opt = torch.optim.Adam(dann_C.parameters(), lr=1e-3)
    D_opt = torch.optim.Adam(dann_D.parameters(), lr=1e-3)

    hook = DANNHook(opts=[G_opt, C_opt, D_opt])

    print("--- Performing Domain Adaptation with DANN ---")

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
                # the discriminator must predict which domain a feature came
                # src domain features will be represented by 0 (label 0)
                # target domain features will be represented by 1 (label 1)
                "src_domain": torch.zeros(
                    src_imgs.size(0), dtype=torch.long, device=device
                ),
                "target_domain": torch.ones(
                    target_imgs.size(0), dtype=torch.long, device=device
                ),
            }
            models = {"G": dann_G, "C": dann_C, "D": dann_D}

            _, losses = hook({**models, **batch})

        print(f"Epoch {epoch+1}/{num_epochs}:")
        pprint(losses)

        print("-- Source domain --")
        test(dann_G, dann_C, device, mnist_loader)

        print("-- Target domain --")
        test(dann_G, dann_C, device, usps_loader)


if __name__ == "__main__":
    main()
