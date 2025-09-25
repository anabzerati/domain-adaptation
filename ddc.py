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

from itertools import cycle
from pprint import pprint
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def test(G, C, device, test_loader):
    G.eval()
    C.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            features = G(data)
            output = C(features)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n"
    )


def train(
    G,
    C,
    dataloader,
    optimizer,
    criterion,
    device,
    num_epochs=5,
    save_path="mnist_split.pth",
):
    """
    Train G (feature extractor) + C (classifier) for multiple epochs and save weights at the end.

    Args:
        G: feature extractor (nn.Module)
        C: classifier (nn.Module)
        dataloader: DataLoader with (images, labels)
        optimizer: torch optimizer (with params from G + C)
        criterion: loss function (e.g., nn.CrossEntropyLoss())
        device: "cuda" or "cpu"
        num_epochs: number of epochs to train
        save_path: where to save the trained weights
    """
    for epoch in range(num_epochs):
        G.train()
        C.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(dataloader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            features = G(imgs)
            outputs = C(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

    torch.save({"G": G.feature_extractor.state_dict(), "C": C.state_dict()}, save_path)
    print(f"Model saved to {save_path}")


"""
Pytorch adapt uses G network as the feature extractor
"""


class G_class(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = CNN()

    def forward(self, x):
        return self.feature_extractor(x, return_embedding=True)


""" 
Pytorch adapt uses a C network as classification
"""


class C_class(nn.Module):
    def __init__(self, in_features=500, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


def main():

    # dados
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
    G = G_class().to(device)
    C = C_class().to(device)

    save_path = "mnist_split.pth"
    if os.path.exists(save_path):
        # checkpoint = torch.load(save_path, map_location=device)
        # G.feature_extractor.load_state_dict(checkpoint["G"])
        # C.load_state_dict(checkpoint["C"])
        pass
    else:
        # pre training on mnist
        optimizer = torch.optim.Adam(list(G.parameters()) + list(C.parameters()), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        train(
            G,
            C,
            mnist_loader,
            optimizer,
            criterion,
            device,
            num_epochs=5,
            save_path="mnist_split.pth",
        )

    G_opt = torch.optim.Adam(G.parameters(), lr=1e-4)
    C_opt = torch.optim.Adam(C.parameters(), lr=1e-4)

    # -----------------------------
    # Hook MMD according to original paper Deep Domain Confusion
    # -----------------------------
    kernel_scales = get_kernel_scales(low=-3, high=3, num_kernels=10)
    loss_fn = MMDLoss(kernel_scales=kernel_scales, mmd_type="quadratic")

    lambda_weight = 0.1

    # AlignerPlusCHook combines a classification loss with a customized loss. In the original implementation of DDC technique
    # the loss function is the classification loss combined with the quadratic MMD.
    #
    # Default classification loss function is Cross Entropy Loss

    hook = AlignerPlusCHook(opts=[G_opt, C_opt], loss_fn=loss_fn)

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

    models = {"G": G, "C": C}
    print("Test USPS before domain adaptation")
    test(G, C, device, usps_loader)
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
            models = {"G": G, "C": C}

            _, losses = hook({**models, **batch})

        # Exibir perdas médias por época
        print(f"Epoch {epoch+1}/{num_epochs}:")
        pprint(losses)

        test(G, C, device, usps_loader)

if __name__ == "__main__":
    main()