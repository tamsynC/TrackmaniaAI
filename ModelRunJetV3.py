import torch
import torch.nn as nn
import numpy as np
import cv2
import mss
import win32gui
from torchvision import transforms
import time
import keyboard
import segmentation_models_pytorch as smp

# Model parameters (same as training)
MODEL_PATH = 'unet_model.pth'
NUM_CLASSES = 5
INPUT_SIZE = (512, 512)
NUM_ACTIONS = 4
ACTIONS = ['d', 'a', 'w', 's']  # right, left, forward, backward

# Labels
LABEL_BACKGROUND = 0
LABEL_CAR = 1
LABEL_TRACK = 4
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_segmentation_model(path):
    """Load the segmentation model"""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def find_trackmania_window():
    """Find Trackmania window"""
    def enum_windows_callback(hwnd, window_list):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "Trackmania" in title:
                window_list.append(hwnd)
    
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    if not windows:
        raise Exception("Trackmania window not found. Make sure Trackmania is running!")
    
    hwnd = windows[0]
    rect = win32gui.GetWindowRect(hwnd)
    return {"top": rect[1], "left": rect[0], "width": rect[2] - rect[0], "height": rect[3] - rect[1]}

class QNetwork(nn.Module):
    """Q-Network architecture (same as training)"""
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=6, stride=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Calculate the actual size
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_output = self.conv(dummy_input)
            linear_input_size = conv_output.view(1, -1).size(1)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(linear_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def segment_frame(model, frame):
    """Segment the frame using the segmentation model"""
    resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    transform = transforms.ToTensor()
    input_tensor = transform(resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    return mask, resized

def calculate_track_direction(mask):
    """Calculate which direction the car should turn based on track layout"""
    car_mask = (mask == LABEL_CAR)
    track_mask = (mask == LABEL_TRACK)
    
    # Find car center
    car_ys, car_xs = np.where(car_mask)
    if len(car_xs) == 0:
        return 0
    car_center_x = int(np.mean(car_xs))
    
    # Find track pixels
    track_ys, track_xs = np.where(track_mask)
    if len(track_xs) == 0:
        return 0
    
    # Count track pixels on left vs right of car
    left_track = np.sum(track_xs < car_center_x)
    right_track = np.sum(track_xs > car_center_x)
    
    if left_track + right_track == 0:
        return 0
    
    # Return direction: positive = more track on right, negative = more track on left
    direction = (right_track - left_track) / (left_track + right_track)
    return np.clip(direction, -1, 1)

def preprocess_frame(frame):
    """Preprocess frame for neural network input"""
    transform = transforms.ToTensor()
    if len(frame.shape) == 3 and frame.shape[2] == 3:  # HWC format
        tensor = transform(frame).float()  # Convert to CHW format
        return tensor.numpy()
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

def select_action(q_net, state):
    """Select action using the trained Q-network"""
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = q_net(state_t)
        action_probs = torch.sigmoid(q_values.squeeze())
        
        # Convert Q-values to binary actions
        action_idx = [
            int(action_probs[0].item() > 0.5),  # Right
            int(action_probs[1].item() > 0.5),  # Left  
            int(action_probs[2].item() > 0.4),  # Forward
            int(action_probs[3].item() > 0.6),  # Backward
        ]
    
    return action_idx

def execute_action(action_idx):
    """Execute the action by pressing/releasing keys"""
    for i, should_press in enumerate(action_idx):
        if should_press:
            keyboard.press(ACTIONS[i])
        else:
            keyboard.release(ACTIONS[i])

def release_all_keys():
    """Release all keys"""
    for key in ACTIONS:
        keyboard.release(key)

def main():
    print("🔧 Loading models...")
    
    # Load segmentation model
    try:
        seg_model = load_segmentation_model(MODEL_PATH)
        print("✅ Segmentation model loaded!")
    except Exception as e:
        print(f"❌ Failed to load segmentation model: {e}")
        return
    
    # Load Q-network
    try:
        q_net = QNetwork((3, INPUT_SIZE[0], INPUT_SIZE[1]), NUM_ACTIONS).to(device)
        
        # Try to load a trained model (adjust filename as needed)
        model_files = [
            'trackmania_dqn_final_optimized.pth'
        ]
        
        model_loaded = False
        for model_file in model_files:
            try:
                q_net.load_state_dict(torch.load(model_file, map_location=device))
                print(f"✅ Q-network loaded from {model_file}!")
                model_loaded = True
                break
            except:
                continue
        
        if not model_loaded:
            print("⚠️ No trained model found. The AI will act randomly.")
        
        q_net.eval()
        
    except Exception as e:
        print(f"❌ Failed to load Q-network: {e}")
        return
    
    # Find Trackmania window
    try:
        monitor = find_trackmania_window()
        print(f"🎮 Found Trackmania window: {monitor}")
    except Exception as e:
        print(f"❌ {e}")
        return
    
    print("\n🚗 Starting AI driver...")
    print("Press 'q' to quit")
    print("Press 'r' to restart track (Backspace)")
    
    try:
        with mss.mss() as sct:
            while True:
                # Check for quit command
                if keyboard.is_pressed('q'):
                    print("🛑 Quitting...")
                    break
                
                # Check for restart command
                if keyboard.is_pressed('r'):
                    print("🔄 Restarting track...")
                    release_all_keys()
                    time.sleep(0.1)
                    keyboard.press_and_release('backspace')
                    time.sleep(2)
                    continue
                
                try:
                    # Capture screen
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)[..., :3]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Segment frame
                    mask, resized_frame = segment_frame(seg_model, frame)
                    
                    # Preprocess for Q-network
                    state = preprocess_frame(resized_frame)
                    
                    # Calculate track direction (for debugging)
                    track_direction = calculate_track_direction(mask)
                    
                    # Select action using AI
                    if model_loaded:
                        action_idx = select_action(q_net, state)
                    else:
                        # Random action if no model loaded
                        action_idx = [
                            0 if track_direction > 0.1 else 1 if track_direction < -0.1 else 0,  # Turn based on track
                            1 if track_direction < -0.1 else 0,  # Left
                            1,  # Always forward
                            0   # Never backward
                        ]
                    
                    # Execute action
                    execute_action(action_idx)
                    
                    # Print current action (optional - comment out if too spammy)
                    action_names = []
                    for i, pressed in enumerate(action_idx):
                        if pressed:
                            action_names.append(['Right', 'Left', 'Forward', 'Backward'][i])
                    
                    if action_names:
                        print(f"🎮 Action: {', '.join(action_names)} | Track Dir: {track_direction:.2f}")
                    
                    time.sleep(0.05)  # Small delay to prevent overwhelming the system
                    
                except Exception as e:
                    print(f"⚠️ Frame processing error: {e}")
                    time.sleep(0.1)
                    continue
    
    except KeyboardInterrupt:
        print("🛑 Interrupted by user")
    
    finally:
        print("🧹 Cleaning up...")
        release_all_keys()
        print("✅ Done!")

if __name__ == "__main__":
    main()