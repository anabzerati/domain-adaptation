import torch
from tqdm import tqdm
from pytorch_adapt.hooks import AlignerPlusCHook
from pytorch_adapt.layers import CORALLoss
from itertools import cycle
from pprint import pprint
import os
from ddc_resnet50 import test, train, G, C
from adaptiode import sim2real

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    source_loader, target_loader = sim2real(32, 8, use_imagenet_norm=True)

    # The network must be divided on a feature extractor G and a classification head G
    coral_G = G().to(device)
    coral_C = C().to(device)

    G_opt = torch.optim.Adam(coral_G.parameters(), lr=1e-4)
    C_opt = torch.optim.Adam(coral_C.parameters(), lr=1e-4)

    # AlignPlusCHook aligns a classification loss with a customized loss function
    # In this case it will align the CrossEntropyLoss with CORALLoss
    hook = AlignerPlusCHook(opts=[G_opt, C_opt], loss_fn=CORALLoss())
    
    print("Testing pretrained model (Imagenet) before DA algorithm...")
    test(coral_G, coral_C, device, target_loader)

    print("Performing Domain Adaptation with Deep Coral")

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
            }
            models = {"G": coral_G, "C": coral_C}

            _, losses = hook({**models, **batch}) # uses hook to optimize weights with customized loss

        # Epoch loss
        print(f"Epoch {epoch+1}/{num_epochs}:")
        pprint(losses)

        # Uses target data for testing
        test(coral_G, coral_C, device, target_loader) 

if __name__ == "__main__":
    main()