import cv2
import mss
import numpy as np
import keyboard
import pygetwindow as gw
import time
import os

# === Configuration ===
WINDOW_TITLE = "Trackmania"  # Change this to match the game window title
SAVE_DIR = "trackmania_frames_2"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_window_bbox(title):
    win = gw.getWindowsWithTitle(title)[0]
    return {'top': win.top, 'left': win.left, 'width': 1920, 'height': 1200}

def main():
    bbox = get_window_bbox(WINDOW_TITLE)
    print("Press 'X' to start recording...")

    while not keyboard.is_pressed('x'):
        time.sleep(0.1)

    print("Recording started! Press ESC to stop.")

    with mss.mss() as sct:
        i = 0
        while True:
            if keyboard.is_pressed('esc'):
                print("Recording stopped.")
                break

            img = np.array(sct.grab(bbox))
            frame_path = os.path.join(SAVE_DIR, f"frame_{i:05d}.jpg")
            cv2.imwrite(frame_path, img)
            i += 1
            time.sleep(1/60)  # Capture at ~60 fps

if __name__ == "__main__":
    main()
