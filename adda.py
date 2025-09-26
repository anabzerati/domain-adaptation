import copy
from pprint import pprint
from tqdm import tqdm
from itertools import cycle

import torch
from torch import nn
from pytorch_adapt.hooks import ADDAHook
from pytorch_adapt.models import Discriminator

from ddc_resnet50 import train, test, G, C
from adaptiode import sim2real, sim2real_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# data
source_train_loader, source_test_loader, target_loader = sim2real_split(32, 8, use_imagenet_norm=True)

# models
adda_G = G().to(device)
adda_C = C().to(device)
adda_D = Discriminator().to(device)

G_opt = torch.optim.Adam(adda_G.parameters())
C_opt = torch.optim.Adam(adda_C.parameters())
D_opt = torch.optim.Adam(adda_D.parameters())

print("--- Training model on source data ---")
# training feature extractor and classifier
# need to optimize both model's parameters
optimizer = torch.optim.Adam(list(adda_G.parameters()) + list(adda_C.parameters()), lr = 0.0001)
criterion = nn.CrossEntropyLoss()

train(adda_G, adda_C, source_train_loader, source_test_loader, optimizer, criterion, device)

print("--- Testing pretrained model on target data ---")
test(adda_G, adda_C, device, target_loader)

print("--- Performing Domain Adaptation with ADDA ---")

# target model
# begins as a copy of source's feature extractor
T = copy.deepcopy(adda_G).to(device)
T_opt = torch.optim.Adam(T.parameters())

hook = ADDAHook(g_opts=[T_opt], d_opts=[D_opt])

# all data to perform the adaptation
source_loader, target_loader = sim2real()

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
                # the discriminator must predict which domain a feature came from
                # src domain features will be represented by 0 (label 0)
                # target domain features will be represented by 1 (label 1)
                "src_domain": torch.zeros(
                    src_imgs.size(0), dtype=torch.long, device=device
                ),
                "target_domain": torch.ones(
                    target_imgs.size(0), dtype=torch.long, device=device
                ),
            }

            models = {"G": adda_G, "C": adda_C, "D": adda_D, "T": T}

            _, losses = hook({**models, **batch})

        # epoch loss
        print(f"Epoch {epoch+1}/{num_epochs}:")
        pprint(losses)

        # target data for testing
        test(T, adda_C, device, target_loader) 
