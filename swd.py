import copy
from pprint import pprint
from tqdm import tqdm
from itertools import cycle

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_adapt.hooks import MCDHook
from pytorch_adapt.layers import MultipleModels, SlicedWasserstein
from pytorch_adapt.utils import common_functions as c_f

from ddc import G_class as G, C_class as C

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
    swd_C = C().to(device)

    # MCD needs 2 classifiers
    C_ = MultipleModels(swd_C, c_f.reinit(copy.deepcopy(swd_C))).to(device)

    G_opt = torch.optim.Adam(swd_G.parameters(), lr=1e-4)
    C_opt = torch.optim.Adam(C_.parameters(), lr=1e-4)

    print("--- Training model on source data ---")
    # training feature extractor and classifiers
    optimizer = torch.optim.Adam(list(swd_G.parameters()) + list(C_.parameters()), lr = 0.0001)
    criterion = nn.CrossEntropyLoss()

    train_source(swd_G, C_, mnist_loader, device, optimizer, criterion)

    print("--- Testing pretrained model on target data ---")
    print(f"Acc={evaluate(swd_G, C_, usps_loader, device)}")


    print("Performing Domain Adaptation with sliced wasserstein discrepancy")

    # Wasserstein distance as loss 
    loss_fn = SlicedWasserstein(m=128)
    hook = MCDHook(g_opts=[G_opt], c_opts=[C_opt], discrepancy_loss_fn=loss_fn)

    num_epochs = 10
    for epoch in range(num_epochs):
        swd_G.train()
        C_.train()

        usps_iter = iter(cycle(usps_loader))
        for src_imgs, src_labels in tqdm(mnist_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            target_imgs, _ = next(usps_iter)

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            target_imgs = target_imgs.to(device)

            batch = {
                "src_imgs": src_imgs,
                "src_labels": src_labels,
                "target_imgs": target_imgs,
            }

            models = {"G": swd_G, "C": C_}
            _, losses = hook({**models, **batch})

        # Epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        pprint(losses)

        print(f"Target USPS accuracy:\n") 
        print(f"C[0] = {evaluate(swd_G, C_, usps_loader, device, mode='first')}")
        print(f"C[1] = {evaluate(swd_G, C_, usps_loader, device, mode='second')}")
        print(f"Both = {evaluate(swd_G, C_, usps_loader, device, mode='both')}")

def train_source(G_, C_, dataloader, device, optimizer, criterion, epochs=5):
    for epoch in range(epochs):
        G_.train()
        C_.train()
        total_loss = 0

        for imgs, labels in tqdm(dataloader, desc=f"[Source Pretrain] Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)

            feats = G_(imgs)
            out1 = C_.models[0](feats)
            out2 = C_.models[1](feats)

            # average output
            outputs = (out1 + out2) / 2.0

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(G_, C_, dataloader, device, mode="both")
        print(f"Epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}, acc={acc:.2%}")


# Because SWD trains 2 classifiers, we need a different evaluation
# mode = "first" -> tests only on C.models[0]
# mode = "second" ->  C.models[1]
# mode = "both" -> average of C.models (default)
def evaluate(G, C, dataloader, device, mode='both'):
    G.eval()
    C.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            feats = G(imgs)
            out1 = C.models[0](feats)
            out2 = C.models[1](feats)

            if mode == "first":
                outputs = out1
            elif mode == "second":
                outputs = out2
            elif mode == "both": # average output
                outputs = (out1 + out2) / 2.0
            else:
                raise ValueError("mode must be 'first', 'second' ou 'both'")

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total 

if __name__ == "__main__":
    main()
