import copy
from pprint import pprint
from tqdm import tqdm
from itertools import cycle

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_adapt.hooks import MCDHook
from pytorch_adapt.layers import MultipleModels, SlicedWasserstein
from pytorch_adapt.utils import common_functions as c_f

from ddc import train, test, G_class as G, C_class as C

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
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

    mnist_loader = DataLoader(mnist, batch_size=32, shuffle=True)
    usps_loader = DataLoader(usps, batch_size=32, shuffle=True)

    # The network must be divided on a feature extractor G and a classification head C
    swd_G = G().to(device)
    swd_C = C()

    # MCD needs 2 classifiers
    C_ = MultipleModels(swd_C, c_f.reinit(copy.deepcopy(swd_C))).to(device)

    G_opt = torch.optim.Adam(swd_G.parameters(), lr=1e-4)
    C_opt_ = torch.optim.Adam(C_.parameters(), lr=1e-4)

    # Wasserstein distance as loss 
    loss_fn = SlicedWasserstein(m=128)
    hook = MCDHook(g_opts=[G_opt], c_opts=[C_opt_], discrepancy_loss_fn=loss_fn)
    
    print("Testing pretrained model (Imagenet) before DA algorithm...")
    test(swd_G, swd_C, device, usps_loader)

    print("Performing Domain Adaptation with sliced wasserstein discrepancy")

    num_epochs = 10
    for epoch in range(num_epochs):
        source_iter = iter(mnist_loader)
        target_iter = iter(cycle(usps_loader))

        for _ in tqdm(range(len(mnist_loader))):
            src_imgs, src_labels = next(source_iter)
            target_imgs, _ = next(target_iter) # doesn't need target labels

            src_imgs = src_imgs.to(device)
            src_labels = src_labels.to(device)
            target_imgs = target_imgs.to(device)

            batch = {
                "src_imgs": src_imgs,
                "src_labels": src_labels,
                "target_imgs": target_imgs,
            }
            models = {"G": swd_G, "C": C_}

            _, losses = hook({**models, **batch}) # uses hook to optimize weights with customized loss

        # Epoch loss
        print(f"Epoch {epoch+1}/{num_epochs}:")
        pprint(losses)

        # Uses target data for testing
        test(swd_G, swd_C, device, usps_loader) 

if __name__ == "__main__":
    main()