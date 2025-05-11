import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CNNAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # [B, 64, H/4, W/4]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# [B, 128, H/8, W/8]
            nn.ReLU(),
            nn.Flatten(),  # [B, 128 * H/8 * W/8]
        )

        self.latent_dim = latent_dim
        self.encoder_fc = nn.Linear(128 * 8 * 8, latent_dim)  # Assuming input images are 64x64

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # [B, 64, H/4, W/4]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # [B, 3, H, W]
            nn.Sigmoid(),  # Normalize output to [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.encoder_fc(z)
        x_hat = self.decoder_fc(z).view(-1, 128, 8, 8)
        x_hat = self.decoder(x_hat)
        return x_hat, z  # Return both reconstruction and latent vector
