# Get representative checkpoint images (maybe 10–50)
# Manually place them in a folder like: "checkpoint_samples/"

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from CNNAutoencoder import CNNAutoencoder
from unlabledImageDataset import UnlabeledImageDataset

# dataset = UnlabeledImageDataset("checkpoint_samples", transform=transforms)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
dataset = UnlabeledImageDataset("checkpoint_samples", transform=transform)


loader = DataLoader(dataset, batch_size=1)

model = CNNAutoencoder()

embeddings = []
for img, _ in loader:
    # img = img.cuda()
    with torch.no_grad():
        _, z = model(img)
    embeddings.append(z.squeeze())

torch.save(embeddings, "checkpoint_embeddings.pt")
