import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

mnist = datasets.MNIST('data', train=True)
svhn = datasets.SVHN('data', split="train")

# transformando imagens em uma lista de intensidades de nivel de cinza 
flattened_pixels = []
for img, _ in svhn:
    img_array = np.array(img)
    flattened_pixels.extend(img_array.flatten())

pixel_data = np.array(flattened_pixels)

# histograma para extrair a PDF
hist_counts, bin_edges = np.histogram(pixel_data, bins=256, range=(0, 256), density=True)

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist_counts, width=1, color='skyblue')
plt.title('Probability Distribution Function of Pixel Intensities')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()