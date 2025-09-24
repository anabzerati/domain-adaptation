import torch
from tqdm import tqdm
from model import CNN
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_adapt.hooks import AlignerPlusCHook
from pytorch_adapt.layers import MMDLoss
from pytorch_adapt.layers.utils import get_kernel_scales
from pprint import pprint
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
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = CNN()
    def forward(self, x):
        return self.feature_extractor(x, return_embedding=True)

class C(nn.Module):
    def __init__(self, in_features=500, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.fc(x)
    
device = "cuda" if torch.cuda.is_available() else "cpu"

# dados
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
])

transform_svhn = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

mnist = datasets.MNIST('data', train=True, download=True, transform=transform_mnist)
svhn = datasets.SVHN('data', split="train", download=True, transform=transform_svhn)


mnist_loader = DataLoader(mnist, batch_size=32, shuffle=True)
svhn_loader = DataLoader(svhn, batch_size=32, shuffle=True)

G = G()
C = C()
G_opt = torch.optim.Adam(G.parameters(), lr=1e-3)
C_opt = torch.optim.Adam(C.parameters(), lr=1e-3)

# -----------------------------
# Hook MMD
# -----------------------------
kernel_scales = get_kernel_scales(low=-3, high=3, num_kernels=10)
loss_fn = MMDLoss(kernel_scales=kernel_scales, mmd_type= "quadratic")

hook = AlignerPlusCHook(opts=[G_opt, C_opt], loss_fn=loss_fn)

# -----------------------------
# Loop de treino (exemplo 1 passo)
# -----------------------------
mnist_iter = iter(mnist_loader)
svhn_iter = iter(svhn_loader)

src_imgs, src_labels = next(mnist_iter)
target_imgs, _ = next(svhn_iter)

batch = {
    "src_imgs": src_imgs,
    "src_labels": src_labels,
    "target_imgs": target_imgs,
}

models = {"G": G, "C": C}
test(G,C,device,svhn_loader)
# _, losses = hook({**models, **batch})
# pprint(losses)

num_epochs = 5
for epoch in range(num_epochs):
    mnist_iter = iter(mnist_loader)
    svhn_iter = iter(svhn_loader)
    
    for _ in tqdm(range(min(len(mnist_loader), len(svhn_loader)))):
        src_imgs, src_labels = next(mnist_iter)
        target_imgs, _ = next(svhn_iter)
        
        batch = {
            "src_imgs": src_imgs,
            "src_labels": src_labels,
            "target_imgs": target_imgs
        }
        models = {"G": G, "C": C}
        
        _, losses = hook({**models, **batch})
    
    # Exibir perdas médias por época
    print(f"Epoch {epoch+1}/{num_epochs}:")
    pprint(losses)
    
    test(G,C,device,svhn_loader)
