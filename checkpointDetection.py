import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from torchvision import transforms
import mss
import win32gui
import win32con
import win32process
import psutil

# Configuration
MODEL_PATH = 'unet_model.pth'
NUM_CLASSES = 5
CLASS_COLORS = np.array([
    [0, 0, 0],        # background
    [255, 0, 0],      # class 1
    [0, 255, 0],      # class 2 (Finish line)
    [0, 0, 255],      # class 3 (Checkpoint)
    [255, 255, 0],    # class 4
])
CLASS_NAMES = ["background", "class_1", "finish_line", "checkpoint", "class_4"]

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

# Return list of detected class names
def get_detected_classes(mask):
    detected_classes = []
    unique_classes = np.unique(mask)
    for class_id in unique_classes:
        if isinstance(class_id, (np.integer, int)):
            class_id = int(class_id)
            if class_id < len(CLASS_NAMES):
                detected_classes.append(CLASS_NAMES[class_id])
    return detected_classes

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

# Live segmentation loop without OpenCV display
def live_segment(model):
    monitor = find_trackmania_window()
    print(f"Tracking Trackmania window at: {monitor}")
    checkpoint_count = 0
    total_checkpoints = 8
    finish_line_cleared = False

    # with mss.mss() as sct:
    #     while True:
    #         sct_img = sct.grab(monitor)
    #         frame = np.array(sct_img)[..., :3]  # RGB
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #         resized_frame = cv2.resize(frame, (512, 512))  # Resize to match training size
    #         mask = segment_frame(model, resized_frame)
    #         detected_classes = get_detected_classes(mask)

    #         # Track checkpoint clearance
    #         if "checkpoint" in detected_classes:
    #             checkpoint_count += 1
    #             print(f"[INFO] Checkpoint {checkpoint_count}/{total_checkpoints} cleared.")
    #             # prevent duplicate prints for the same frame
    #             cv2.waitKey(1000)  # short delay

    #         # Track finish line clearance
    #         if "finish_line" in detected_classes and not finish_line_cleared:
    #             print("[INFO] Finish line cleared!")
    #             finish_line_cleared = True

    #         if checkpoint_count >= total_checkpoints:
    #             print("[INFO] All checkpoints cleared!")
    #             break

    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             print("[INFO] Manual quit.")
    #             break

    checkpoint_visible_last_frame = False
    finish_line_visible_last_frame = False
    
    with mss.mss() as sct:
        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)[..., :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame, (512, 512))
    
            mask = segment_frame(model, resized_frame)
            detected_classes = get_detected_classes(mask)
    
            # --- Checkpoint transition detection ---
            checkpoint_visible_now = "checkpoint" in detected_classes
            if checkpoint_visible_now and not checkpoint_visible_last_frame:
                checkpoint_count += 1
                print(f"[INFO] Checkpoint {checkpoint_count}/{total_checkpoints} cleared.")
            elif not checkpoint_visible_now and checkpoint_visible_last_frame:
                print(f"[INFO] Checkpoint {checkpoint_count} no longer visible.")
            checkpoint_visible_last_frame = checkpoint_visible_now
    
            # --- Finish line detection ---
            finish_line_visible_now = "finish_line" in detected_classes
            if finish_line_visible_now and not finish_line_visible_last_frame and not finish_line_cleared:
                print("[INFO] Finish line cleared!")
                finish_line_cleared = True
            finish_line_visible_last_frame = finish_line_visible_now
    
            if checkpoint_count >= total_checkpoints:
                print("[INFO] All checkpoints cleared!")
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Manual quit.")
                break


if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    live_segment(model)
