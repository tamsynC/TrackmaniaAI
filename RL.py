import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import mss
import win32gui
from torchvision import transforms
import time

# ===== CONFIGURATION =====
MODEL_PATH = 'unet_model.pth'
NUM_CLASSES = 5
INPUT_SIZE = (512, 512)  # Resize input to this resolution

# Class IDs (must match training)
LABEL_BACKGROUND = 0
LABEL_CAR = 1
LABEL_TRACK = 4
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3

CLASS_NAMES = ["background", "car", "checkpoint", "finish", "track"]
CLASS_COLORS = np.array([
    [0, 0, 0],        # background - black
    [255, 0, 0],      # car - red
    [0, 255, 0],      # checkpoint - green
    [0, 0, 255],      # finish - blue
    [255, 255, 0],    # track - yellow
], dtype=np.uint8)

# ===== Model Loading =====
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

# ===== Capture Trackmania Window =====
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

# ===== Preprocessing & Segmentation =====
transform = transforms.ToTensor()

def segment_frame(model, frame):
    resized = cv2.resize(frame, INPUT_SIZE)
    input_tensor = transform(resized).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return mask, resized

# ===== Visual Overlay =====
def overlay_mask(image, mask):
    color_mask = CLASS_COLORS[mask]
    overlayed = cv2.addWeighted(image, 0.6, color_mask, 0.4, 0)
    return overlayed

# ===== Utility: Bounding Box Detection =====
def get_bbox(binary_mask):
    ys, xs = np.where(binary_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (min(xs), min(ys), max(xs), max(ys))

def boxes_overlap(a, b):
    if a is None or b is None:
        return False
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

# ===== Visual Debug: Draw BBoxes =====
def draw_bbox(image, bbox, color, label=""):
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def detect_events(mask, vis_image):
    car_mask = (mask == LABEL_CAR)
    track_mask = (mask == LABEL_TRACK)
    checkpoint_mask = (mask == LABEL_CHECKPOINT)
    finish_mask = (mask == LABEL_FINISH)

    car_bbox = get_bbox(car_mask)
    track_bbox = get_bbox(track_mask)
    finish_bbox = get_bbox(finish_mask)

    # Visual debugging
    draw_bbox(vis_image, car_bbox, (255, 0, 0), "Car")
    draw_bbox(vis_image, track_bbox, (0, 255, 0), "Track")
    draw_bbox(vis_image, finish_bbox, (0, 255, 255), "Finish")

    events = []

    if not boxes_overlap(car_bbox, track_bbox):
        events.append("❌ Car is out of bounds!")

    # Cluster checkpoint blobs using connected components
    checkpoint_mask_uint8 = checkpoint_mask.astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(checkpoint_mask_uint8)

    checkpoint_crossed = False
    for label in range(1, num_labels):  # skip background label 0
        component_mask = (labels_im == label)
        bbox = get_bbox(component_mask)
        draw_bbox(vis_image, bbox, (0, 0, 255), f"Checkpoint {label}")

        if boxes_overlap(car_bbox, bbox):
            checkpoint_crossed = True

    if checkpoint_crossed:
        events.append("✅ Checkpoint crossed!")

    # Optionally check finish line overlap
    # if boxes_overlap(car_bbox, finish_bbox):
    #     events.append("🏁 Finish line crossed!")

    return events



# # ===== Event Detection =====
# def detect_events(mask, vis_image):
#     car_mask = (mask == LABEL_CAR)
#     track_mask = (mask == LABEL_TRACK)
#     checkpoint_mask = (mask == LABEL_CHECKPOINT)
#     finish_mask = (mask == LABEL_FINISH)

#     car_bbox = get_bbox(car_mask)
#     track_bbox = get_bbox(track_mask)
#     checkpoint_bbox = get_bbox(checkpoint_mask)
#     finish_bbox = get_bbox(finish_mask)

#     # Visual debugging
#     draw_bbox(vis_image, car_bbox, (255, 0, 0), "Car")
#     draw_bbox(vis_image, track_bbox, (0, 255, 0), "Track")
#     draw_bbox(vis_image, checkpoint_bbox, (0, 0, 255), "Checkpoint")
#     draw_bbox(vis_image, finish_bbox, (0, 255, 255), "Finish")

#     events = []

#     if not boxes_overlap(car_bbox, track_bbox):
#         events.append("❌ Car is out of bounds!")

#     if boxes_overlap(car_bbox, checkpoint_bbox):
#         events.append("✅ Checkpoint crossed!")

#     # if boxes_overlap(car_bbox, finish_bbox):
#     #     events.append("🏁 Finish line crossed!")

#     return events

# ===== Main Loop =====
def watch_trackmania(model):
    monitor = find_trackmania_window()
    print(f"🎮 Watching Trackmania window at: {monitor}")
    with mss.mss() as sct:
        try:
            while True:
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)[..., :3]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mask, resized_frame = segment_frame(model, frame)
                vis = overlay_mask(resized_frame, mask)
                events = detect_events(mask, vis)

                if events:
                    print("EVENTS:", "; ".join(events))

                cv2.imshow("Trackmania Live Segmentation", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("🛑 Monitoring stopped.")
        cv2.destroyAllWindows()

# ===== Entry Point =====
if __name__ == '__main__':
    print("🔧 Loading model...")
    model = load_model(MODEL_PATH)
    print("👀 Starting live segmentation...")
    watch_trackmania(model)