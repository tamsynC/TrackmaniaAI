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
import threading
from queue import Queue
import win32api
import win32con

# Model and game configuration
MODEL_PATH = 'unet_model.pth'  # Segmentation model path
DQN_MODEL_PATH = 'trackmania_dqn_interrupted.pth'  # Your trained DQN model
NUM_CLASSES = 5
INPUT_SIZE = (256, 256)

ACTIONS = ['a', 'd', 'w', 's']
NUM_ACTIONS = len(ACTIONS)

# Segmentation labels
LABEL_BACKGROUND = 0
LABEL_CAR = 1  
LABEL_TRACK = 4
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3

# Control variables
action_queue = Queue()
control_thread = None
stop_control_thread = False
current_action = [0, 0, 0, 0]
action_lock = threading.Lock()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Transform for preprocessing
transform = transforms.ToTensor()

class QNetwork(nn.Module):
    """Same Q-Network architecture as in training"""
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Calculate conv output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_output = self.conv(dummy_input)
            linear_input_size = conv_output.view(1, -1).size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
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

def load_dqn_model(path, input_shape, num_actions):
    """Load the trained DQN model"""
    model = QNetwork(input_shape, num_actions).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def find_trackmania_window():
    """Find the Trackmania window"""
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
    """Segment the frame using the segmentation model"""
    resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    
    with torch.no_grad():
        input_tensor = transform(resized).unsqueeze(0).to(device)
        output = model(input_tensor)
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    return mask, resized

    
def preprocess_frame(frame):

    if len(frame.shape) == 3 and frame.shape[2] == 3:  
        tensor = transform(frame).float()  
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    
    return tensor.numpy()

def get_centroid(binary_mask):
    """Get centroid of a binary mask"""
    ys, xs = np.where(binary_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))

# def calculate_track_direction(mask):
#     """Calculate track direction for the car"""
#     car_mask = (mask == LABEL_CAR)
#     track_mask = (mask == LABEL_TRACK)
    
#     car_center = get_centroid(car_mask)
#     if car_center is None:
#         return 0
    
#     track_ys, track_xs = np.where(track_mask)
#     if len(track_xs) == 0:
#         return 0
    
#     # Sample track points for direction calculation
#     sample_size = min(len(track_xs), len(track_xs) // 10 + 1)
#     indices = np.random.choice(len(track_xs), sample_size, replace=False)
#     sampled_xs = track_xs[indices]
    
#     car_x = car_center[0]
#     left_weight = np.sum(sampled_xs < car_x)
#     right_weight = np.sum(sampled_xs > car_x)
    
#     if left_weight + right_weight == 0:
#         return 0
    
#     direction = (right_weight - left_weight) / (left_weight + right_weight)
#     return np.clip(direction, -1, 1)


def select_action(q_net, state):

    # Optimized Q-network inference
    with torch.no_grad():
        # Create tensor on CPU, then move to GPU
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        state_tensor = state_tensor.to(device, non_blocking=True)
        
        # Forward pass
        q_values = q_net(state_tensor)
        action_probs = torch.sigmoid(q_values.squeeze())
        print(action_probs)
        # Move back to CPU for processing
        action_probs_cpu = action_probs.cpu()
        
        # Clear GPU tensors immediately
        del state_tensor, q_values, action_probs
        
        # Process on CPU
        action = [0] * 4
        
        # Your existing action logic using action_probs_cpu
        left_prob = action_probs_cpu[0].item()
        right_prob = action_probs_cpu[1].item()
        
        if left_prob > 0.98 and right_prob > 0.98:
            if left_prob > right_prob:
                action[0] = 1
            else:
                action[1] = 1
        else:
            if left_prob > 0.98:
                action[0] = 1
            if right_prob > 0.98:
                action[1] = 1
        
        w_prob = action_probs_cpu[2] if len(action_probs_cpu) > 2 else 0
        s_prob = action_probs_cpu[3] if len(action_probs_cpu) > 3 else 0
        
        if w_prob > 0.98 or s_prob > 0.98:
            if w_prob > s_prob:
                action[2] = 1
            else:
                action[3] = 1
        
        del action_probs_cpu
        return action

    

def start_control_thread():
    """Start the dedicated control thread"""
    global control_thread, stop_control_thread
    
    stop_control_thread = False
    control_thread = threading.Thread(target=control_thread_worker, daemon=True)
    control_thread.start()
    print("🎮 Control thread started")

def stop_control_thread_func():
    """Stop the dedicated control thread"""
    global control_thread, stop_control_thread
    
    stop_control_thread = True
    if control_thread and control_thread.is_alive():
        control_thread.join(timeout=1.0)
    force_release_all_keys()
    print("🛑 Control thread stopped")

def control_thread_worker():
    """Fixed control thread worker that maintains key states"""
    global stop_control_thread, current_action

    prev_action_state = [False] * len(ACTIONS)
    currently_pressed = [False] * len(ACTIONS)

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

            # Copy current action
            with action_lock:
                action_to_execute = current_action.copy()

            # Handle key presses and releases
            for i, should_press in enumerate(action_to_execute):
                key = ACTIONS[i]
                
                # If we should press and we're not currently pressing
                if should_press and not currently_pressed[i]:
                    keyboard.press(key)
                    currently_pressed[i] = True
                    print(f"🔥 {key} pressed and held")
                
                # If we should NOT press and we ARE currently pressing
                elif not should_press and currently_pressed[i]:
                    keyboard.release(key)
                    currently_pressed[i] = False
                    print(f"🔄 {key} released")

            # Small sleep to avoid tight loop
            time.sleep(0.01)  # 100Hz update rate

        except Exception as e:
            print(f"Control thread error: {e}")
            time.sleep(0.1)

    # Release all keys when stopping
    for i, key in enumerate(ACTIONS):
        if currently_pressed[i]:
            keyboard.release(key)
            currently_pressed[i] = False
    print("🛑 All keys released on thread stop")


def force_release_all_keys():
    """Force release all keys using both keyboard lib and Windows API"""
    for key in ACTIONS:
        try:
            # Use keyboard library first
            keyboard.release(key)
        except:
            pass
        
        try:
            # Also use Windows API as backup
            vk_code = ord(key.upper())
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
        except:
            pass
    time.sleep(0.01)


def press_action(action_list):
    """Queue action for the dedicated control thread"""
    global action_queue

    try:
        # Clear old actions from queue
        while not action_queue.empty():
            try:
                action_queue.get_nowait()
                action_queue.task_done()
            except:
                break

        # Add new action
        action_queue.put_nowait(action_list.copy())
    except Exception as e:
        print(f"Error queuing action: {e}")

def main():
    print("🚗 Trackmania AI Driver - Inference Mode")
    print("Press 'q' to quit")
    
    try:
        # Load models
        print("🔧 Loading segmentation model...")
        seg_model = load_segmentation_model(MODEL_PATH)
        
        print("🧠 Loading DQN model...")
        dqn_model = load_dqn_model(DQN_MODEL_PATH, (3, INPUT_SIZE[0], INPUT_SIZE[1]), NUM_ACTIONS)
        
        # Start control thread
        start_control_thread()
        
        # Find Trackmania window
        monitor = find_trackmania_window()
        print(f"🎮 Found Trackmania window: {monitor}")
        
        print("🏁 Starting AI driver... (Press 'q' to quit)")
        
        with mss.mss() as sct:
            frame_count = 0
            
            while True:
                # Check for quit command
                if keyboard.is_pressed('q'):
                    print("🛑 Quit command received")
                    break
                
                try:
                    # Capture frame
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)[..., :3]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Segment frame and resize
                    mask, resized = segment_frame(seg_model, frame)

                    # Prepare state for DQN
                    state = preprocess_frame(resized)

                    # Get action
                    action = select_action(dqn_model, state)

                    # Queue the action for execution
                    press_action(action)

                    frame_count += 1

                    
                    # Execute action
                    press_action(action)
                    print(action)
                    # Print status occasionally
                    # frame_count += 1
                    # if frame_count % 30 == 0:  # Every ~0.5 seconds at 60fps
                    #     action_str = "".join([
                    #         "L" if action[0] else "",
                    #         "R" if action[1] else "", 
                    #         "F" if action[2] else "",
                    #         "B" if action[3] else ""
                    #     ])
                    #     if not action_str:
                    #         action_str = "None"
                    #     print(f"🎮 Frame {frame_count} - Action: {action_str}, Track Dir: {track_direction:.2f}")
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.1)  # ~60 FPS
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    time.sleep(0.1)
                    continue
    
    except KeyboardInterrupt:
        print("🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        stop_control_thread_func()
        force_release_all_keys()
        print("🏁 AI driver stopped")

if __name__ == "__main__":
    main()