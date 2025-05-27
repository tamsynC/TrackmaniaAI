import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Configuration
MODEL_PATH = 'C:/Users/tamsy/Documents/TrackmaniaAI/unet_model.pth'   # Path to the trained model
NUM_CLASSES = 5                  # Number of classes used during training
CLASS_COLORS = np.array([
    [125, 0, 125],    # background
    [255, 0, 0],      # class 1
    [0, 255, 0],      # class 2
    [0, 0, 255],      # class 3
    [255, 255, 0],    # class 4
])

CLASS_NAMES = ['background', 'car', 'checkpoint', 'finish', 'track']

# Load trained model
def load_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Segment an image
def segment_image(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    return image, mask

# def print_detected_features(mask):
#     unique_classes = np.unique(mask)
#     print("Detected features:")
#     for class_id in enumerate(unique_classes):
#         class_id = int(class_id)
#         if class_id < len(CLASS_NAMES):
#             print("f{CLASS_NAMES[class_id]} (id: {class_id})")

def print_detected_features(mask):
    unique_classes = np.unique(mask)
    print("Detected features in the frame:")
    for class_id in unique_classes:
        if isinstance(class_id, (np.integer, int)):
            class_id = int(class_id)
            if class_id < len(CLASS_NAMES):
                print(f" - {CLASS_NAMES[class_id]} (id: {class_id})")
            else:
                print(f" - Unknown class (id: {class_id})")
    

# Visualize segmentation
def visualize_segmentation(image, mask):
    mask_color = CLASS_COLORS[mask]
    overlay = cv2.addWeighted(image, 0.6, mask_color.astype(np.uint8), 0.4, 0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Overlay")
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

# Main usage example
if __name__ == '__main__':
    image_path = 'Trackmania_Coco_Segmentation/test/frame_00183_jpg.rf.a4a3245b681b5f87065077aadf7946c6.jpg'  
    model = load_model(MODEL_PATH)
    image, mask = segment_image(model, image_path)
    print_detected_features(mask)
    visualize_segmentation(image, mask)
