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
import win32api
import win32con

# Model parameters (must match training)
MODEL_PATH_SEGMENTATION = 'unet_model.pth'
MODEL_PATH_DQN = 'trackmania_dqn_checkpoint_150.pth'
NUM_CLASSES = 5
INPUT_SIZE = (512, 512)

ACTIONS = ['a', 'd', 'w', 's']  # left, right, forward, backward
NUM_ACTIONS = len(ACTIONS)

# Labels from your training
LABEL_BACKGROUND = 0
LABEL_CAR = 1
LABEL_TRACK = 4
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class QNetwork(nn.Module):
    """Same Q-Network architecture as training"""
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

def load_dqn_model(path):
    """Load the trained DQN model"""
    model = QNetwork((3, INPUT_SIZE[0], INPUT_SIZE[1]), NUM_ACTIONS).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def find_trackmania_window():
    """Find Trackmania window for screen capture"""
    def enum_windows_callback(hwnd, window_list):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "Trackmania" in title:
                window_list.append(hwnd)
    
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    if not windows:
        raise Exception("Trackmania window not found. Make sure Trackmania is running.")
    
    hwnd = windows[0]
    rect = win32gui.GetWindowRect(hwnd)
    return {"top": rect[1], "left": rect[0], "width": rect[2] - rect[0], "height": rect[3] - rect[1]}

def segment_frame(model, frame):
    """Segment the frame to get track information"""
    resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    transform = transforms.ToTensor()
    input_tensor = transform(resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    return mask, resized

def calculate_track_direction(mask):
    """Calculate which direction the track is turning"""
    car_mask = (mask == LABEL_CAR)
    track_mask = (mask == LABEL_TRACK)
    
    car_center = get_centroid(car_mask)
    if car_center is None:
        return 0
    
    track_ys, track_xs = np.where(track_mask)
    if len(track_xs) == 0:
        return 0
    
    car_x = car_center[0]
    left_weight = np.sum(track_xs < car_x)
    right_weight = np.sum(track_xs > car_x)
    
    if left_weight + right_weight == 0:
        return 0
    
    direction = (right_weight - left_weight) / (left_weight + right_weight)
    return np.clip(direction, -1, 1)

def get_centroid(binary_mask):
    """Get centroid of a binary mask"""
    ys, xs = np.where(binary_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))

def preprocess_frame(frame):
    """Preprocess frame for the neural network"""
    transform = transforms.ToTensor()
    if len(frame.shape) == 3 and frame.shape[2] == 3:  # HWC format
        tensor = transform(frame).float()  # This converts to CHW format
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    
    return tensor.numpy()

def select_action(q_net, state, track_direction, confidence_threshold=0.3):
    """Select action using the trained model"""
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = q_net(state_t)
        
        # Convert to probabilities for each action independently
        action_probs = torch.sigmoid(q_values.squeeze())
        
        # Apply thresholds for each action (you may need to tune these)
        action_idx = [
            (action_probs[0] > 0.4).int().item(),  # Left
            (action_probs[1] > 0.4).int().item(),  # Right  
            (action_probs[2] > 0.3).int().item(),  # Forward
            (action_probs[3] > 0.6).int().item(),  # Backward
        ]
        
        # Print action probabilities for debugging
        print(f"Action probs: L:{action_probs[0]:.2f} R:{action_probs[1]:.2f} F:{action_probs[2]:.2f} B:{action_probs[3]:.2f} -> {action_idx}")
        
    return action_idx

def press_action(action_idx):
    """Press the keys corresponding to the action"""
    for i in range(len(action_idx)):
        if action_idx[i] == 1:
            keyboard.press(ACTIONS[i])
        else:
            keyboard.release(ACTIONS[i])

def release_all_keys():
    """Release all keys"""
    for key in ACTIONS:
        keyboard.release(key)

def force_release_all_keys():
    """Force release all keys using Windows API"""
    for key in ACTIONS:
        vk_code = ord(key.upper())
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.1)

def main():
    print("🔧 Loading models...")
    
    try:
        # Load models
        seg_model = load_segmentation_model(MODEL_PATH_SEGMENTATION)
        dqn_model = load_dqn_model(MODEL_PATH_DQN)
        print("✅ Models loaded successfully!")
        
        # Find Trackmania window
        monitor = find_trackmania_window()
        print(f"🎮 Found Trackmania window: {monitor}")
        
        print("\n🚗 Starting AI driver...")
        print("Press 'q' to quit, 'r' to restart track")
        
        with mss.mss() as sct:
            step_count = 0
            
            while True:
                try:
                    # Check for quit key
                    if keyboard.is_pressed('q'):
                        print("🛑 Quitting...")
                        break
                    
                    # Check for restart key
                    if keyboard.is_pressed('r'):
                        print("🔄 Restarting track...")
                        force_release_all_keys()
                        keyboard.press_and_release('backspace')
                        time.sleep(3)
                        continue
                    
                    # Capture screen
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)[..., :3]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Segment frame
                    mask, resized_frame = segment_frame(seg_model, frame)
                    
                    # Calculate track direction
                    track_direction = calculate_track_direction(mask)
                    
                    # Preprocess for DQN
                    state = preprocess_frame(resized_frame)
                    
                    # Select action
                    action_idx = select_action(dqn_model, state, track_direction)
                    
                    # Execute action
                    press_action(action_idx)
                    
                    # Status update every 100 steps
                    step_count += 1
                    if step_count % 100 == 0:
                        print(f"🏃 Step {step_count} - Track direction: {track_direction:.2f}")
                    
                    # Control frame rate
                    time.sleep(1/30)  # ~30 FPS
                    
                except Exception as e:
                    print(f"⚠️ Error in main loop: {e}")
                    continue
                    
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("Make sure both 'unet_model.pth' and 'trackmania_dqn_final_optimized.pth' are in the current directory.")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Ensure all keys are released
        force_release_all_keys()
        print("🏁 AI driver stopped.")

if __name__ == "__main__":
    main()