import torch
import torch.nn as nn
import numpy as np
import mss
import cv2
from torchvision import transforms
import pygetwindow as gw
import time
import keyboard  # replaces pynput
from PIL import Image

from pynput.keyboard import Controller, Key
keyboard_controller = Controller()


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(17024, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cnn(x)

# Map to actual keyboard keys
keys_map = ['a', 'd', 'w', 's']
threshold = 0.45
pressed_keys = set()  # track pressed keys

def get_window_bbox(title):
    win = gw.getWindowsWithTitle(title)[0]
    return {'top': win.top, 'left': win.left, 'width': 1920, 'height': 1200}

def main():
    model = SimpleCNN()
    model.load_state_dict(torch.load("model_behavioral_cloning_final.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor()
    ])

    bbox = get_window_bbox("Trackmania")

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
                        keyboard_controller.press(key)
                        print(f"Pressing {keys_map[i]} ({val:.2f})")
                        pressed_keys.add(key)
                else:
                    if key in pressed_keys:
                        keyboard_controller.release(key)
                        # print(f"Released: {key}")
                        pressed_keys.remove(key)

            time.sleep(1/30)

if __name__ == "__main__":
    main()
