import cv2  # opencv
import mss  # screenshot module
import numpy as np
import keyboard
import pygetwindow as gw  # to obtain GUI info on an application
import time
import os
import json

WINDOW_TITLE = "Trackmania"  # Adjust this to your actual window title
SAVE_DIR = "training_data"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(SAVE_DIR, "log.json")

data_log = []

def get_window_bbox(title):
    win = gw.getWindowsWithTitle(title)[0]
    return {'top': win.top, 'left': win.left, 'width': 1920, 'height':1200}

def get_keys():
    keys = ['a', 'd', 'w', 's']
    return [int(keyboard.is_pressed(k)) for k in keys]

def main():
    bbox = get_window_bbox(WINDOW_TITLE)
    print("Press 'X' to start recording...")

    # Wait for 'x' key press
    while not keyboard.is_pressed('x'):
        time.sleep(0.1)

    print("Starting capture... Press ESC to stop.")

    with mss.mss() as sct:
        i = 0
        while True:
            if keyboard.is_pressed('esc'):
                print("Stopping capture.")
                break

            img = np.array(sct.grab(bbox))
            frame_name = f"frame_{i:05d}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR, frame_name), img)

            keys_pressed = get_keys()
            data_log.append({
                'frame': frame_name,
                'keys': keys_pressed
            })

            i += 1
            time.sleep(1/30)  # ~30 fps

    with open(LOG_FILE, 'w') as f:
        json.dump(data_log, f, indent=2)

if __name__ == "__main__":
    main()
