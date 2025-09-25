""" 
File to handle adaptiode dataset (folder data2)

The dataset is proposed on the paper
https://openaccess.thecvf.com/content/WACV2021/papers/Ringwald_Adaptiope_A_Modern_Benchmark_for_Unsupervised_Domain_Adaptation_WACV_2021_paper.pdf

It creates a modern benchmark for domain adaptation, providing

synthetic -> real
real -> synthetic 


All classes are balanced
"""

from torchvision import datasets, transforms
from typing import Tuple
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

def sim2real(
    batch_size: int = 32,
    num_workers: int = 8,
    use_imagenet_norm: bool = False,
) -> Tuple[DataLoader, DataLoader]:    
    """
    Prepares domain adaptation datasets:
      - Source (synthetic): all data
      - Target (real_life): all data

    Returns:    use_imagenet_norm: bool = True,

        Tuple containing:
          - source_loader (DataLoader)
          - target_loader (DataLoader)
    """
    data_dir = "data2"

    if use_imagenet_norm:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    source =  datasets.ImageFolder("data2/synthetic", transform=transform)

    target = datasets.ImageFolder("data2/real_life", transform=transform)

    source_loader = DataLoader(source, batch_size, shuffle="True", num_workers=num_workers)
    
    target_loader = DataLoader(target, batch_size, shuffle="True", num_workers=num_workers)


    return source_loader, target_loader



def sim2real_split(
    batch_size: int = 32,
    num_workers: int = 2,
    source_split: float = 0.8,
    use_imagenet_norm: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepares domain adaptation datasets when there is no train/test split:
      - Source (synthetic): split on the fly into train + test
      - Target (real_life): all used as test only

    Args:
        batch_size: batch size for DataLoaders
        num_workers: number of DataLoader workers
        source_split: fraction of source dataset to use for training

    Returns:
        Tuple containing:
          - source_train_loader (DataLoader)
          - source_test_loader (DataLoader)
          - target_test_loader (DataLoader)
    """

    if use_imagenet_norm:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    source =  datasets.ImageFolder("data2/synthetic", transform=transform)

    target = datasets.ImageFolder("data2/real_life", transform=transform)


    n_total = len(source)
    n_train = int(n_total * source_split)
    n_test = n_total - n_train
    source_train, source_test = random_split(source, [n_train, n_test])


    source_train_loader = DataLoader(
        source_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    source_test_loader = DataLoader(
        source_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    target_test_loader = DataLoader(
        target, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return source_train_loader, source_test_loader, target_test_loader

def show_dataloader_samples(loader, classes=None, num_images=8):
    """
    Display a few images and their labels from a PyTorch DataLoader.

    Args:
        loader: PyTorch DataLoader
        classes: list of class names (optional)
        num_images: number of images to show
    """
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # Mostrar no máximo num_images
    images = images[:num_images]
    labels = labels[:num_images]

    plt.figure(figsize=(12, 4))

    for i in range(len(images)):
        img = images[i]

        # converter CxHxW -> HxWxC
        if img.shape[0] == 1:  # grayscale
            img = img.squeeze(0)
        else:
            img = img.permute(1, 2, 0)

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img.detach().cpu().numpy(), cmap='gray' if img.ndim == 2 else None)
        plt.axis('off')
        if classes:
            plt.title(classes[labels[i].item()])
        else:
            plt.title(str(labels[i].item()))

    plt.show()
