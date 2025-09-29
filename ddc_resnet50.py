
import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from pytorch_adapt.hooks import AlignerPlusCHook
from pytorch_adapt.layers import MMDLoss
from pytorch_adapt.layers.utils import get_kernel_scales 
from pprint import pprint
import os
from adaptiode import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def test(G, C, device, test_loader):
    G.eval()
    C.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"[Testing]", leave=False): 
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
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return accuracy


def train(G, C, train_loader, val_loader, optimizer, criterion, device, num_epochs=5, save_path="best_model.pth"):
    """
    Train G (feature extractor) + C (classifier) and save the best model based on validation accuracy.
    
    Args:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        ... (other arguments remain the same)
    """
    print("--- Starting Training ---")
    best_val_acc = 0.0  # Keep track of the best validation accuracy

    for epoch in range(num_epochs):
        G.train()
        C.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            features = G(imgs)
            outputs = C(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pred = outputs.argmax(dim=1)
            correct_train += pred.eq(labels).sum().item()
            total_train += labels.size(0)

        avg_loss = running_loss / total_train
        train_acc = 100.0 * correct_train / total_train
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

        val_acc = test(G, C, device, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best model found! Accuracy: {val_acc:.2f}%. Saving model to {save_path}")
            torch.save({
                "G": G.state_dict(),
                "C": C.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc,
            }, save_path)
    
    print("--- Training Finished ---")
    print(f"Best validation accuracy achieved: {best_val_acc:.2f}%")


"""
Pytorch adapt uses G network as the feature extractor

We are using ResNet50 as backbone
"""

class G(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # removes last fc layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        # Flatten the features from [batch_size, 2048, 1, 1] to [batch_size, 2048]
        return torch.flatten(features, 1)


""" 
Pytorch adapt uses a C network as classification
"""

class C(nn.Module):
    def __init__(self, in_features=2048, num_classes=123):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


def main():
    # datasets
    source, target = sim2real(32, 8, use_imagenet_norm=True)

    # The network must be divided on a feature extractor G and a classification head G
    G_ddc = G().to(device)
    C_ddc = C().to(device)

    G_opt = torch.optim.Adam(G_ddc.parameters(), lr=1e-4)
    C_opt = torch.optim.Adam(C_ddc.parameters(), lr=1e-4)

    # -----------------------------
    # Hook MMD according to original paper Deep Domain Confusion
    # -----------------------------
    kernel_scales = get_kernel_scales(low=-3, high=3, num_kernels=10)

    # AlignerPlusCHook combines a classification loss with a customized loss. In the original implementation of DDC technique
    # the loss function is the classification loss combined with the quadratic MMD.
    # Default classification loss function is Cross Entropy Loss
    hook = AlignerPlusCHook(
        opts=[G_opt, C_opt],
        loss_fn=MMDLoss(kernel_scales=kernel_scales, mmd_type="quadratic")
    )

    print("Testing pretrained model before DA algorithm...")
    test(G_ddc, C_ddc, device, target)


    print("Performing Domain Adaptation with DDC")

    num_epochs = 10
    for epoch in range(num_epochs):
        source_iter = iter(source)
        target_iter = iter(target)

        for _ in tqdm(range(min(len(source), len(target)))):

            src_imgs, src_labels = next(source_iter)
            target_imgs, _ = next(target_iter) # DDC doesn't need target labels

            src_imgs = src_imgs.to(device)
            src_labels = src_labels.to(device)
            target_imgs = target_imgs.to(device)

            batch = {
                "src_imgs": src_imgs,
                "src_labels": src_labels,
                "target_imgs": target_imgs,
            }
            models = {"G": G_ddc, "C": C_ddc}

            _, losses = hook({**models, **batch}) # uses hook to optimize weights with customized loss

        # Epoch loss
        print(f"Epoch {epoch+1}/{num_epochs}:")
        pprint(losses)

        # Uses target data for testing
        test(G_ddc, C_ddc, device, target) 

if __name__ == "__main__":
    main()