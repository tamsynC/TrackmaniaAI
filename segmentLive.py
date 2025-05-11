import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import mss
import win32gui
import win32con
import win32process
import psutil

# Configuration
MODEL_PATH = 'unet_model.pth'   # Path to the trained model
NUM_CLASSES = 5                  # Number of classes used during training
CLASS_COLORS = np.array([
    [0, 0, 0],        # background
    [255, 0, 0],      # class 1
    [0, 255, 0],      # class 2
    [0, 0, 255],      # class 3
    [255, 255, 0],    # class 4
])
CLASS_NAMES = ["background", "class_1", "class_2", "class_3", "class_4"]

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

# Segment a frame
def segment_frame(model, frame):
    transform = transforms.ToTensor()
    input_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return mask

# Print detected features
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

# Overlay mask on frame
def overlay_mask(frame, mask):
    color_mask = CLASS_COLORS[mask]
    overlay = cv2.addWeighted(frame, 0.6, color_mask.astype(np.uint8), 0.4, 0)
    return overlay

# Find the bounding box of the Trackmania window
def find_trackmania_window():
    def enum_windows_callback(hwnd, window_list):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "Trackmania" in title:
                window_list.append(hwnd)
    
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    if not windows:
        raise Exception("Trackmania window not found.")

    hwnd = windows[0]
    rect = win32gui.GetWindowRect(hwnd)
    return {"top": rect[1], "left": rect[0], "width": rect[2] - rect[0], "height": rect[3] - rect[1]}

# Live segmentation from game window
def live_segment(model):
    monitor = find_trackmania_window()
    print(f"Tracking Trackmania window at: {monitor}")
    with mss.mss() as sct:
        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)[..., :3]  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            resized_frame = cv2.resize(frame, (512, 512))  # Resize to match training size
            mask = segment_frame(model, resized_frame)
            result = overlay_mask(resized_frame, mask)

            print_detected_features(mask)

            cv2.imshow("Trackmania Live Segmentation", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    live_segment(model)
