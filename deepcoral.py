import torch
from tqdm import tqdm
from model import CNN, train
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_adapt.hooks import AlignerPlusCHook
from pytorch_adapt.layers import MMDLoss
from pytorch_adapt.layers.utils import get_kernel_scales

from pytorch_adapt.layers import CORALLoss
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
    coral_G = G().to(device)
    coral_C = C().to(device)

    save_path = "mnist_split.pth"
    if os.path.exists(save_path):
        # checkpoint = torch.load(save_path, map_location=device)
        # coral_G.feature_extractor.load_state_dict(checkpoint["G"])
        # coral_C.load_state_dict(checkpoint["C"])
        pass
    else:
        # pre training on mnist
        optimizer = torch.optim.Adam(
            list(coral_G.parameters()) + list(C.parameters()), lr=1e-3
        )
        criterion = nn.CrossEntropyLoss()
        train(
            coral_G,
            coral_C,
            mnist_loader,
            optimizer,
            criterion,
            device,
            num_epochs=5,
            save_path="mnist_split.pth",
        )

    G_opt = torch.optim.Adam(coral_G.parameters(), lr=1e-4)
    C_opt = torch.optim.Adam(coral_C.parameters(), lr=1e-4)

    # AlignPlusCHook aligns a classification loss with a customized loss function
    # In this case it will align the CrossEntropyLoss with CORALLoss
    hook = AlignerPlusCHook(opts=[G_opt, C_opt], loss_fn=CORALLoss())
    # -----------------------------
    # Loop de treino (exemplo 1 passo)
    # -----------------------------
    mnist_iter = iter(mnist_loader)
    usps_iter = iter(usps_loader)

    src_imgs, src_labels = next(mnist_iter)
    target_imgs, _ = next(usps_iter)

    batch = {
        "src_imgs": src_imgs,
        "src_labels": src_labels,
        "target_imgs": target_imgs,
    }

    models = {"G": coral_G, "C": coral_C}
    print("Test USPS before domain adaptation")
    test(coral_G, coral_C, device, usps_loader)
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
            }
            models = {"G": coral_G, "C": coral_C}

            _, losses = hook({**models, **batch})

        # Exibir perdas médias por época
        print(f"Epoch {epoch+1}/{num_epochs}:")
        pprint(losses)

        test(coral_G, coral_C, device, usps_loader)


if __name__ == "__main__":
    main()
