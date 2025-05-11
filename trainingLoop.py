import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from CNNAutoencoder import CNNAutoencoder
import torch.nn as nn
from torchvision.datasets import DatasetFolder
from PIL import Image
from unlabledImageDataset import UnlabeledImageDataset

# Dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((64, 64)),
    transforms.ToTensor()
])


dataset = UnlabeledImageDataset("trackmania_frames", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, optimizer, loss
model = CNNAutoencoder(latent_dim=128) #add .cuda() for gpu
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, _ in dataloader:
        # x = x.cuda() #for gpu
        x_hat, _ = model(x)
        loss = loss_fn(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "cnn_autoencoder.pth")
print("Model saved to cnn_autoencoder.pth")

    # Optional: visualize reconstruction
    # model.eval()
    # with torch.no_grad():
    #     x_sample = next(iter(dataloader))[0][:8] #add .cuda() for gpu
    #     recon, _ = model(x_sample)
    #     grid = make_grid(torch.cat([x_sample, recon], dim=0), nrow=8)
    #     plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    #     plt.title(f"Reconstructions - Epoch {epoch+1}")
    #     plt.axis('off')
    #     plt.show()
