# import torch
# import torch.nn as nn
# import numpy as np
# import mss
# import cv2
# from torchvision import transforms
# import pygetwindow as gw
# import time
# import keyboard  # Replaces pynput for key press simulation
# from PIL import Image

# # Load the trained model
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
#             nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(17024, 128),
#             nn.ReLU(),
#             nn.Linear(128, 4),
#             nn.Sigmoid()  # Apply sigmoid activation for output probabilities
#         )

#     def forward(self, x):
#         return self.cnn(x)

# # Map to actual keyboard keys
# keys_map = ['a', 'd', 'w', 's']
# threshold = 0.45  # Threshold to decide when to press/release keys
# pressed_keys = set()  # Track currently pressed keys

# def get_window_bbox(title):
#     # Get the bounding box of the Trackmania window
#     try:
#         win = gw.getWindowsWithTitle(title)[0]
#         return {'top': win.top, 'left': win.left, 'width': 1920, 'height': 1200}
#     except IndexError:
#         print("Trackmania window not found!")
#         return None

# def main():
#     # Load the trained model
#     model = SimpleCNN()
#     model.load_state_dict(torch.load("model_behavioral_cloning_final.pth"))
#     model.eval()  # Set model to evaluation mode

#     # Image transformation
#     transform = transforms.Compose([
#         transforms.Resize((120, 160)),
#         transforms.ToTensor()
#     ])

#     # Get Trackmania window bounding box
#     bbox = get_window_bbox("Trackmania")
#     if bbox is None:
#         return

#     # Start screen capture
#     with mss.mss() as sct:
#         while True:
#             # Capture the screen using mss
#             img = np.array(sct.grab(bbox))
#             frame = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

#             # Transform the captured frame for model input
#             input_tensor = transform(Image.fromarray(frame)).unsqueeze(0)

#             # Predict key presses using the model
#             with torch.no_grad():
#                 output = model(input_tensor)[0]
#                 print("Model output:", [round(val.item(), 2) for val in output])

#             # Check model output and press/release keys accordingly
#             for i, val in enumerate(output):
#                 key = keys_map[i]
#                 if val > threshold:  # If probability is above threshold, press the key
#                     if key not in pressed_keys:
#                         keyboard.press(key)
#                         print(f"Pressing {key} ({val:.2f})")
#                         pressed_keys.add(key)
#                 else:  # If probability is below threshold, release the key
#                     if key in pressed_keys:
#                         keyboard.release(key)
#                         print(f"Releasing {key}")
#                         pressed_keys.remove(key)

#             # Sleep to maintain a consistent framerate
#             time.sleep(1/30)

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
import numpy as np
import mss
import cv2
from torchvision import transforms
import pygetwindow as gw
import time
import keyboard
from PIL import Image

# Define the matching model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        print("Initializing SimpleCNN model")
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(17024, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 4, bias=True),
            nn.Sigmoid()
        )
        print("Model initialized")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Map to actual keyboard keys
keys_map = ['a', 'd', 'w', 's']
threshold = 0.45
pressed_keys = set()

def get_window_bbox(title):
    try:
        win = gw.getWindowsWithTitle(title)[0]
        return {'top': win.top, 'left': win.left, 'width': 1920, 'height': 1200}
    except IndexError:
        print("Trackmania window not found!")
        return None

def main():
    model = SimpleCNN()
    model.load_state_dict(torch.load("model_behavioral_cloning_finalv2.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor()
    ])

    bbox = get_window_bbox("Trackmania")
    if bbox is None:
        return

    with mss.mss() as sct:
        while True:
            img = np.array(sct.grab(bbox))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            input_tensor = transform(Image.fromarray(frame)).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)[0]
                print("Model output:", [round(val.item(), 2) for val in output])

            for i, val in enumerate(output):
                key = keys_map[i]
                if val > threshold:
                    if key not in pressed_keys:
                        keyboard.press(key)
                        print(f"Pressing {key} ({val:.2f})")
                        pressed_keys.add(key)
                else:
                    if key in pressed_keys:
                        keyboard.release(key)
                        print(f"Releasing {key}")
                        pressed_keys.remove(key)

            time.sleep(1 / 30)

if __name__ == "__main__":
    main()
