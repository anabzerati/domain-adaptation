import torch
from tqdm import tqdm
from pytorch_adapt.hooks import AlignerPlusCHook
from pytorch_adapt.layers import CORALLoss
from pytorch_adapt.hooks import MCDHook
from pytorch_adapt.layers import MultipleModels, SlicedWasserstein
from pytorch_adapt.utils import common_functions as c_f
import copy
from itertools import cycle
from pprint import pprint
from ddc_resnet50 import test, G, C
from adaptiode import sim2real

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    source_loader, target_loader = sim2real(32, 8, use_imagenet_norm=True)

    # The network must be divided on a feature extractor G and a classification head C
    swd_G = G().to(device)
    swd_C = C()

    # MCD needs 2 classifiers
    C_ = MultipleModels(C, c_f.reinit(copy.deepcopy(C))).to(device)

    G_opt = torch.optim.Adam(swd_G.parameters(), lr=1e-4)
    C_opt_ = torch.optim.Adam(C_.parameters(), lr=1e-4)

    # Wasserstein distance as loss 
    loss_fn = SlicedWasserstein(m=128)
    hook = MCDHook(g_opts=[G_opt], c_opts=[C_opt_], discrepancy_loss_fn=loss_fn)
    
    print("Testing pretrained model (Imagenet) before DA algorithm...")
    test(swd_G, swd_C, device, target_loader)

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
            models = {"G": G, "C": C_}

            _, losses = hook({**models, **batch}) # uses hook to optimize weights with customized loss

        # Epoch loss
        print(f"Epoch {epoch+1}/{num_epochs}:")
        pprint(losses)

        # Uses target data for testing
        test(swd_G, swd_C, device, target_loader) 

if __name__ == "__main__":
    main()