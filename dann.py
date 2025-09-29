import torch
from tqdm import tqdm
from itertools import cycle
from pprint import pprint

from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.hooks import DANNHook
from pytorch_adapt.models import Discriminator

from ddc_resnet50 import test, G, C
from adaptiode import sim2real


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    source_loader, target_loader = sim2real(32, 8, use_imagenet_norm=True)

    # The network must be divided on a feature extractor G and a classification head G
    dann_G = G().to(device)
    dann_C = C().to(device)

    # The domain adaption component from the original implementation
    # consists of  three (x→1024→1024→1) fully connected layer
    dann_D = Discriminator(in_size=2048, h=1024).to(device)

    models = Models({"G": dann_G, "C": dann_C, "D": dann_D})
    optimizers = Optimizers((torch.optim.Adam, {"lr": 1e-4}))
    optimizers.create_with(models)
    optimizers = list(optimizers.values())

    hook = DANNHook(optimizers)

    print("Testing pretrained model (Imagenet) before DA algorithm...")
    test(dann_G, dann_C, device, source_loader)


    print("Performing Domain Adaptation with DANN")

    num_epochs = 10
    for epoch in range(num_epochs):
        source_iter = iter(cycle(source_loader))
        target_iter = iter(target_loader)

        for _ in tqdm(range(len(target_loader))):
            src_imgs, src_labels = next(source_iter)
            target_imgs, _ = next(target_iter) # doesn't need target labels

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

        # Epoch loss
        print(f"Epoch {epoch+1}/{num_epochs}:")
        pprint(losses)

        # Uses target data for testing
        test(dann_G, dann_C, device, target_loader) 

if __name__ == "__main__":
    main()