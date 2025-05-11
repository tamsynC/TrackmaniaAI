import torch
import cv2
import numpy as np
import mss
import pygetwindow as gw
import keyboard
import time
from torchvision import transforms
from scipy.spatial.distance import cosine

# === Load trained model ===
from CNNAutoencoder import CNNAutoencoder
model = CNNAutoencoder(latent_dim=128)
model.load_state_dict(torch.load("cnn_autoencoder.pth"))
model.eval()

# === Load reference checkpoint embeddings ===
checkpoint_embeddings = torch.load("checkpoint_embeddings.pt")  # List of tensors

# === Parameters ===
SIMILARITY_THRESHOLD = 0.2  # Adjust based on empirical testing
WINDOW_TITLE = "Trackmania"
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
# ])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to RGB mode
    transforms.ToTensor(),
])

def get_window_bbox(title):
    win = gw.getWindowsWithTitle(title)[0]
    return {'top': win.top, 'left': win.left, 'width': 1920, 'height': 1200}

def detect_checkpoint(z, refs, threshold=SIMILARITY_THRESHOLD):
    for ref in refs:
        sim = cosine(z.cpu().detach().numpy(), ref.cpu().detach().numpy())
        if sim < threshold:
            return True
    return False

def main():
    bbox = get_window_bbox(WINDOW_TITLE)
    print("Press X to start monitoring...")

    while not keyboard.is_pressed('x'):
        time.sleep(0.1)

    print("Started! Press ESC to stop.")
    checkpoint_count = 0
    cooldown = 0  # To avoid double-counting same checkpoint

    with mss.mss() as sct:
        while True:
            if keyboard.is_pressed('esc'):
                print("Monitoring stopped.")
                break

            frame = np.array(sct.grab(bbox))
            img = transform(frame).unsqueeze(0)  # [1, 3, 64, 64]

            with torch.no_grad():
                _, z = model(img)

            if cooldown == 0 and detect_checkpoint(z.squeeze(), checkpoint_embeddings):
                print(f"[✓] Checkpoint {checkpoint_count + 1} cleared!")
                checkpoint_count += 1
                cooldown = 15  # Roughly 0.5 seconds if 30fps

            cooldown = max(0, cooldown - 1)
            time.sleep(1/30)

    print(f"Total checkpoints detected: {checkpoint_count}")

if __name__ == "__main__":
    main()
