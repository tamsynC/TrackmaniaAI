import torch
import torch.nn as nn
import numpy as np
import cv2
import mss
import win32gui
from torchvision import transforms
import segmentation_models_pytorch as smp
import time
import keyboard
import threading
from queue import Queue

# Constants from your training script
MODEL_PATH = 'unet_model.pth'
NUM_CLASSES = 5
INPUT_SIZE = (512, 512)
ACTIONS = ['a', 'd', 'w', 's']
NUM_ACTIONS = len(ACTIONS)

# Labels
LABEL_BACKGROUND = 0
LABEL_CAR = 1
LABEL_TRACK = 4
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Action control
action_queue = Queue()
current_action = [0, 0, 0, 0]
action_lock = threading.Lock()
stop_control_thread = False

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape  # c=7 for enhanced state
        
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
        
        # Calculate linear input size
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
    """Load segmentation model"""
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

def segment_frame(model, frame):
    """Segment frame using trained model"""
    resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    transform = transforms.ToTensor()
    input_tensor = transform(resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    return mask, resized

def get_centroid(binary_mask):
    """Get centroid of binary mask"""
    ys, xs = np.where(binary_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))

def calculate_track_direction(mask):
    """Calculate track direction"""
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

def calculate_car_speed(current_pos, prev_pos, dt=1/60):
    """Calculate car speed"""
    if current_pos is None or prev_pos is None:
        return 0
    
    current_pos = np.array(current_pos, dtype=np.float32)
    prev_pos = np.array(prev_pos, dtype=np.float32)
    distance = np.linalg.norm(current_pos - prev_pos)
    speed = distance / dt
    return float(np.clip(speed, 0, 500))

def preprocess_frame(frame):
    """Preprocess frame"""
    transform = transforms.ToTensor()
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        tensor = transform(frame).float()
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    return tensor.numpy()

def create_enhanced_state(resized_frame, mask, car_center, track_direction, car_speed):
    """Create enhanced state with 7 channels"""
    # Base RGB state
    base_state = preprocess_frame(resized_frame)  # Shape: (3, 512, 512)
    
    height, width = INPUT_SIZE
    
    # Additional channels
    direction_channel = np.full((height, width), track_direction, dtype=np.float32)
    speed_normalized = min(car_speed / 100.0, 1.0)
    speed_channel = np.full((height, width), speed_normalized, dtype=np.float32)
    
    # Car position heatmap
    car_channel = np.zeros((height, width), dtype=np.float32)
    if car_center is not None:
        x, y = car_center
        if 0 <= x < width and 0 <= y < height:
            car_channel[max(0, y-10):min(height, y+10), 
                       max(0, x-10):min(width, x+10)] = 1.0
    
    # Mask channel
    mask_channel = (mask.astype(np.float32) / NUM_CLASSES)
    
    # Stack all channels
    enhanced_state = np.stack([
        base_state[0], base_state[1], base_state[2],  # RGB
        direction_channel, speed_channel, car_channel, mask_channel
    ], axis=0)  # Shape: (7, 512, 512)
    
    return enhanced_state

def control_thread_worker():
    """Control thread for smooth key presses"""
    global stop_control_thread, current_action
    
    while not stop_control_thread:
        try:
            # Get new action if available
            try:
                new_action = action_queue.get_nowait()
                with action_lock:
                    current_action = new_action
                action_queue.task_done()
            except:
                pass
            
            # Execute current action
            with action_lock:
                action_to_execute = current_action.copy()
            
            # Press/release keys
            for i, should_press in enumerate(action_to_execute):
                if should_press:
                    keyboard.press(ACTIONS[i])
                else:
                    keyboard.release(ACTIONS[i])
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Control thread error: {e}")
            time.sleep(0.1)
    
    # Release all keys when stopping
    for key in ACTIONS:
        keyboard.release(key)

def press_action(action_idx):
    """Queue action for control thread"""
    try:
        # Clear queue
        while not action_queue.empty():
            try:
                action_queue.get_nowait()
                action_queue.task_done()
            except:
                break
        
        action_queue.put(action_idx.copy())
    except Exception as e:
        print(f"Error queuing action: {e}")

def select_action(q_net, state):
    """Select action using trained model"""
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = q_net(state_t)
        q_values = q_values.squeeze(0)  # Shape becomes [4]
        print(f"Q-values: {q_values.cpu().numpy()}")
        # q_values = torch.argmax(q_values).item()
        # Initialize action list
        action_idx = [0, 0, 0, 0]  # A, D, W, S

        # Threshold checks for W and S (independent)
        action_idx[2] = int(q_values[2].item() > 1300)  # W
        action_idx[3] = int(q_values[3].item() > 120)    # S

        # Thresholds for A and D
        a_val = q_values[0].item()
        d_val = q_values[1].item()
        if -50 > a_val > -56 or -50 > d_val > -56:
            # Only activate one: the higher of A or D
            if a_val > d_val:
                action_idx[0] = 1  # A
            else:
                action_idx[1] = 1  # D

        print(f"Actions: A={action_idx[0]}, D={action_idx[1]}, W={action_idx[2]}, S={action_idx[3]}")

    return action_idx

def main():
    global stop_control_thread
    
    print("🔧 Loading models...")
    
    # Load segmentation model
    try:
        seg_model = load_segmentation_model(MODEL_PATH)
        print("✅ Segmentation model loaded")
    except Exception as e:
        print(f"❌ Error loading segmentation model: {e}")
        return
    
    # Load Q-network
    try:
        q_net = QNetwork((7, INPUT_SIZE[0], INPUT_SIZE[1]), NUM_ACTIONS).to(device)
        
        # Try different possible model file names
        model_files = [
            "tam.pth"

        ]
        
        model_loaded = False
        for model_file in model_files:
            try:
                q_net.load_state_dict(torch.load(model_file, map_location=device))
                print(f"✅ Q-network loaded from {model_file}")
                model_loaded = True
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
                continue
        
        if not model_loaded:
            print("❌ No trained model found! Available files:")
            import os
            for file in os.listdir('.'):
                if file.endswith('.pth'):
                    print(f"  - {file}")
            return
            
    except Exception as e:
        print(f"❌ Error setting up Q-network: {e}")
        return
    
    q_net.eval()
    
    # Start control thread
    print("🎮 Starting control thread...")
    control_thread = threading.Thread(target=control_thread_worker, daemon=True)
    control_thread.start()
    
    # Find game window
    try:
        monitor = find_trackmania_window()
        print(f"🎮 Found Trackmania window: {monitor}")
    except Exception as e:
        print(f"❌ {e}")
        return
    
    print("\n🚗 Starting AI driver...")
    print("Press Ctrl+C to stop")
    
    prev_car_pos = None
    
    with mss.mss() as sct:
        try:
            while True:
                try:
                    # Capture frame
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)[..., :3]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Segment frame
                    mask, resized_frame = segment_frame(seg_model, frame)
                    
                    # Calculate features
                    car_mask = (mask == LABEL_CAR)
                    car_center = get_centroid(car_mask)
                    track_direction = calculate_track_direction(mask)
                    car_speed = calculate_car_speed(car_center, prev_car_pos) if prev_car_pos else 0
                    
                    # Create state
                    state = create_enhanced_state(
                        resized_frame=resized_frame,
                        mask=mask,
                        car_center=car_center,
                        track_direction=track_direction,
                        car_speed=car_speed
                    )
                    
                    # Select and execute action
                    action_idx = select_action(q_net, state)
                    press_action(action_idx)
                    
                    # Update previous position
                    prev_car_pos = car_center
                    
                    # Small delay
                    time.sleep(0.05)
                    
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    time.sleep(0.1)
                    continue
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping AI driver...")
        finally:
            # Cleanup
            stop_control_thread = True
            for key in ACTIONS:
                keyboard.release(key)
            print("✅ AI driver stopped")

if __name__ == "__main__":
    main()