import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from CNNAutoencoder import CNNAutoencoder

def detect_checkpoint(input_tensor, output_tensor, roi_bounds=(20, 60, 20, 60), debug=False):
    """
    Enhanced checkpoint detector using reconstruction error.
    
    Args:
        input_tensor (Tensor): original image tensor of shape (1, 3, H, W)
        output_tensor (Tensor): reconstructed image tensor of shape (1, 3, H, W)
        roi_bounds (tuple): (ymin, ymax, xmin, xmax) ROI bounds for detection
        debug (bool): if True, returns segmentation mask for visualization
    
    Returns:
        bool: True if checkpoint is detected
        (optional) np.ndarray: segmentation mask if debug=True
    """
    input_img = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    recon_img = output_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)

    error = np.abs(input_img - recon_img)
    error_gray = np.mean(error, axis=2)

    # Smooth error map
    error_blur = cv2.GaussianBlur(error_gray, (5, 5), 0)

    # Convert to uint8 for thresholding
    error_uint8 = (error_blur * 255).astype(np.uint8)

    # Adaptive threshold using Otsu
    _, mask = cv2.threshold(error_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # ROI-based detection
    ymin, ymax, xmin, xmax = roi_bounds
    roi = cleaned[ymin:ymax, xmin:xmax]
    detected = roi.mean() > 25  # value from 0-255

    return (detected, cleaned) if debug else detected


# ---------------- Main Script ------------------

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((64, 64)),  
    transforms.ToTensor()
])

# Load image
img_path = "trackmania_frames/frame_00219.jpg"
input_image = Image.open(img_path).convert("RGB")
input_tensor = transform(input_image).unsqueeze(0)  # .cuda()

# Load model
model = CNNAutoencoder(latent_dim=128)
model.load_state_dict(torch.load("cnn_autoencoder.pth"))
# model.cuda()
model.eval()

# Reconstruction
with torch.no_grad():
    output, _ = model(input_tensor)

# Detection with debug output
detected, segmentation_mask = detect_checkpoint(input_tensor, output, debug=True)

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(np.transpose(input_tensor.squeeze().cpu().numpy(), (1, 2, 0)))
axs[0].set_title("Original Image")

axs[1].imshow(np.transpose(output.squeeze().cpu().numpy(), (1, 2, 0)))
axs[1].set_title("Reconstructed Image")

axs[2].imshow(segmentation_mask, cmap="gray")
axs[2].set_title("Segmented Checkpoint Area")

for ax in axs:
    ax.axis("off")

plt.suptitle("Checkpoint Detected" if detected else "No Checkpoint Detected", fontsize=14)
plt.tight_layout()
plt.show()
