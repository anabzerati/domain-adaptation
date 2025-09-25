import torch
from tqdm import tqdm
from model import CNN, train
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from pytorch_adapt.containers import Models, Optimizers

from torch.utils.data import DataLoader
from pytorch_adapt.hooks import DANNHook
from pytorch_adapt.layers import MMDLoss
from pytorch_adapt.layers.utils import get_kernel_scales
from pytorch_adapt.models import Discriminator
from itertools import cycle
from pprint import pprint
import os
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
    # consists of  three (x→1024→1024→1) fully connected layer
    dann_D = Discriminator(in_size=500, h=1024).to(device)

    save_path = "mnist_split.pth"
    if os.path.exists(save_path):
        # checkpoint = torch.load(save_path, map_location=device)
        # dann_G.feature_extractor.load_state_dict(checkpoint["G"])
        # dann_C.load_state_dict(checkpoint["C"])
        pass
    else:
        # pre training on mnist
        optimizer = torch.optim.Adam(
            list(dann_G.parameters()) + list(C.parameters()), lr=1e-3
        )
        criterion = nn.CrossEntropyLoss()
        train(
            dann_G,
            dann_C,
            mnist_loader,
            optimizer,
            criterion,
            device,
            num_epochs=5,
            save_path="mnist_split.pth",
        )

    models = Models({"G": dann_G, "C": dann_C, "D": dann_D})
    optimizers = Optimizers((torch.optim.Adam, {"lr": 1e-4}))
    optimizers.create_with(models)
    optimizers = list(optimizers.values())

    hook = DANNHook(optimizers)
    # -----------------------------

    print("Test USPS before domain adaptation")
    test(dann_G, dann_C, device, usps_loader)
    # test(G,C, device, mnist_loader)

    # _, losses = hook({**models, **batch})
    # pprint(losses)

    num_epochs = 10
    for epoch in range(num_epochs):

        # necessary because datasets have different lengths
        # avoid stopping when run out of data
        usps_iter = iter(cycle(usps_loader))
        mnist_iter = iter(mnist_loader)

        # for _ in tqdm(range(min(len(mnist_loader), len(usps_loader)))):

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

        # Exibir perdas médias por época
        print(f"Epoch {epoch+1}/{num_epochs}:")
        pprint(losses)

        test(dann_G, dann_C, device, usps_loader)


if __name__ == "__main__":
    main()
