import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from ignite.metrics import MaximumMeanDiscrepancy
from ignite.engine import Engine
import torch.nn as nn
from model import CNN  # certifique-se que a CNN implementa return_embedding

device = "cuda" if torch.cuda.is_available() else "cpu"

# dados
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
])

transform_usps = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

mnist = datasets.MNIST('data', train=True, download=True, transform=transform_mnist)
usps = datasets.USPS('data', download=True, transform=transform_usps)
mnist = Subset(mnist, list(range(len(usps))))

# dataset de pares, sendo 1 elemento do mnist e 1 do usps
class PairedDataset(Dataset):
    def __init__(self, ds1, ds2):
        assert len(ds1) == len(ds2)
        self.ds1 = ds1
        self.ds2 = ds2

    def __len__(self):
        return len(self.ds1)

    def __getitem__(self, idx):
        x1, _ = self.ds1[idx]
        x2, _ = self.ds2[idx]
        return x1, x2

paired_dataset = PairedDataset(mnist, usps)
paired_loader = DataLoader(paired_dataset, batch_size=64, shuffle=True, drop_last=True)

# CNN para obter embeddings
model = CNN().to(device)
#model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model.eval()


# engine para MMD em pixels
def eval_step_pixels(engine, batch):
    x1, x2 = batch

    # ajusta dimensões
    x1 = x1.view(x1.size(0), -1).to(device)
    x2 = x2.view(x2.size(0), -1).to(device)

    return x1, x2

evaluator_pixels = Engine(eval_step_pixels)
mmd_pixels = MaximumMeanDiscrepancy()
mmd_pixels.attach(evaluator_pixels, "mmd")

state_pixels = evaluator_pixels.run(paired_loader)
print("MMD entre MNIST e usps - pixels:", state_pixels.metrics["mmd"])

# engine para MMD em embeddings
def eval_step_emb(engine, batch):
    x1, x2 = batch
    x1 = x1.to(device)
    x2 = x2.to(device)

    # extrai embeddings
    with torch.no_grad():
        emb1 = model(x1, return_embedding=True)
        emb2 = model(x2, return_embedding=True)

    return emb1, emb2

evaluator_emb = Engine(eval_step_emb)
mmd_emb = MaximumMeanDiscrepancy()
mmd_emb.attach(evaluator_emb, "mmd")

state_emb = evaluator_emb.run(paired_loader)
print("MMD entre MNIST e usps - embeddings:", state_emb.metrics["mmd"])
