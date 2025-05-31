import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import cv2
import mss
import win32gui
from torchvision import transforms
import torchvision.transforms as T
from torchmetrics.functional import structural_similarity_index_measure as ssim
import time
import random
from collections import deque
import threading
from queue import Queue, Empty
import os
import shutil
import ctypes
from ctypes import wintypes
import win32api
import win32con
import matplotlib.pyplot as plt

import keyboard
import segmentation_models_pytorch as smp
# from prioritized_replay import PrioritizedReplayBuffer, train_step_prioritized

MODEL_PATH = 'unet_model.pth'
NUM_CLASSES = 5
INPUT_SIZE = (512, 512)

ACTIONS = ['a', 'd', 'w', 's']
ACTION_INDEX = [0, 1, 2, 3]
ACTION_WEIGHTS = [0.25, 0.25, 0.9, 0.1]
ACTION_WEIGHTS_NORMALISED = [0.25, 0.25, 0.45, 0.05]
NUM_ACTIONS = len(ACTIONS)

LABEL_BACKGROUND = 0
LABEL_CAR = 1
LABEL_TRACK = 4
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3

# Optimized parameters to reduce computational load
BATCH_SIZE = 64  # Reduced for less GPU memory usage
GAMMA = 0.99
LEARNING_RATE = 3e-4
REPLAY_BUFFER_CAPACITY = 200000    # Reduced buffer size
TARGET_UPDATE_FREQ = 25  # Less frequent updates
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 3000 #change to 50000

# Performance optimizations
EPISODE_TIMEOUT = 60
CRASH_THRESHOLD = 0.87
STUCK_THRESHOLD = 10
CHECKPOINT_CONFIRM_FRAMES = 30
FINISH_CONFIRM_FRAMES = 300  # 5 seconds at 60 FPS
VELOCITY_THRESHOLD = 3
FRAME_SKIP = 8  # Process every 2nd frame to reduce load
TRAIN_FREQUENCY = 2  # Train every 4 steps instead of every step
CHECKPOINT_TIMEOUT = 40  # seconds without checkpoint before penalty
CHECKPOINT_TIMEOUT_PENALTY = -500  # Large penalty for not reaching checkpoint in time


# DISK SPACE MANAGEMENT
CHECKPOINT_SAVE_FREQUENCY = 50  # Save less frequently (every 50 episodes instead of 25)
MAX_CHECKPOINTS_TO_KEEP = 3     # Only keep the 3 most recent checkpoints
MIN_FREE_SPACE_GB = 1.0         # Minimum free space to maintain (in GB)

testingvar = 0




def force_release_all_keys():
    """Force release all keys using Windows API"""
    for key in ACTIONS:
        # Convert to virtual key code
        vk_code = ord(key.upper())
        # Force key up using Windows API
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.1)



def robust_press_action(action_idx):
    """More robust key pressing with state verification"""
    # First, ensure all keys are released
    force_release_all_keys()
    
    # Then press required keys
    for i, pressed in enumerate(action_idx):
        if pressed == 1:
            try:
                keyboard.press(ACTIONS[i])
            except Exception as e:
                print(f"Key press failed for {ACTIONS[i]}: {e}")
                # Fallback to Windows API
                vk_code = ord(ACTIONS[i].upper())
                win32api.keybd_event(vk_code, 0, 0, 0)
def train_step_prioritized(q_net, target_net, optimizer, replay_buffer):
    """Modified training step to work with prioritized replay"""
    if len(replay_buffer) < BATCH_SIZE:
        return

    # Sample from prioritized buffer
    sample_result = replay_buffer.sample(BATCH_SIZE)
    if sample_result is None:
        return
        
    states, actions, rewards, next_states, dones, weights, indices = sample_result

    states_v = torch.FloatTensor(states).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    dones_v = torch.FloatTensor(dones).to(device)
    weights_v = torch.FloatTensor(weights).to(device)

    q_values = q_net(states_v)
    
    with torch.no_grad():
        next_q_values = target_net(next_states_v)
        next_q_max = next_q_values.max(1)[0]
        expected_q_values = rewards_v + (GAMMA * next_q_max * (1 - dones_v))

    state_action_values = (q_values * actions_v).sum(1)
    
    # Calculate TD errors for priority updates
    td_errors = (expected_q_values - state_action_values).detach().cpu().numpy()
    
    # Calculate weighted loss (importance sampling)
    loss = (weights_v * F.mse_loss(state_action_values, expected_q_values, reduction='none')).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    optimizer.step()
    
    # Update priorities in replay buffer
    replay_buffer.update_priorities(indices, td_errors)


# Usage in main() function - replace the ReplayBuffer initialization:
def initialize_prioritized_buffer():
    """How to initialize the prioritized buffer in your main() function"""
    # Replace this line:
    # replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    
    # With this:
    replay_buffer = PrioritizedReplayBuffer(
        capacity=REPLAY_BUFFER_CAPACITY,
        alpha=0.6,      # How much to prioritize (0.6 is good default)
        beta=0.4,       # Importance sampling correction (starts low, increases)
        beta_increment=0.001  # How fast beta increases
    )
    return replay_buffer


# Modified buffer push in main loop - add events parameter:
def example_buffer_push_usage():
    """Example of how to modify your buffer.push() calls"""
    
    # In your main training loop, when you call:
    # replay_buffer.push(prev_state, action_idx, reward, next_state, done)
    
    # Change it to:
    # replay_buffer.push(prev_state, action_idx, reward, next_state, done, events)
    
    # Where 'events' is the list of events from your detect_events() function
    pass


# Additional helper function to analyze buffer priorities (optional)
def analyze_buffer_priorities(replay_buffer):
    """Debug function to see priority distribution"""
    if len(replay_buffer) == 0:
        return
        
    priorities = np.array(replay_buffer.priorities)
    print(f"Priority stats - Min: {priorities.min():.2f}, Max: {priorities.max():.2f}, Mean: {priorities.mean():.2f}")
    print(f"Beta value: {replay_buffer.beta:.3f}")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CODEITERATION5")
print(f"Using device: {device}")

def check_disk_space(path="."):
    """Check available disk space in GB"""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    return free_gb

def cleanup_old_checkpoints(pattern="trackmania_dqn_checkpoint_", keep_count=MAX_CHECKPOINTS_TO_KEEP):
    """Remove old checkpoint files, keeping only the most recent ones"""
    checkpoint_files = []
    for file in os.listdir("."):
        if file.startswith(pattern) and file.endswith(".pth"):
            try:
                # Extract episode number from filename
                episode_num = int(file.replace(pattern, "").replace(".pth", ""))
                checkpoint_files.append((episode_num, file))
            except ValueError:
                continue
    
    # Sort by episode number (newest first)
    checkpoint_files.sort(reverse=True)
    
    # Remove old checkpoints
    for i, (episode_num, filename) in enumerate(checkpoint_files):
        if i >= keep_count:
            try:
                os.remove(filename)
                print(f"🗑️ Removed old checkpoint: {filename}")
            except OSError as e:
                print(f"⚠️ Could not remove {filename}: {e}")

def safe_save_model(model_state_dict, filename):
    """Safely save model with disk space checks"""
    free_space = check_disk_space()
    
    if free_space < MIN_FREE_SPACE_GB:
        print(f"⚠️ Low disk space ({free_space:.2f}GB). Cleaning up old checkpoints...")
        cleanup_old_checkpoints()
        
        # Check again after cleanup
        free_space = check_disk_space()
        if free_space < MIN_FREE_SPACE_GB:
            print(f"❌ Still insufficient disk space ({free_space:.2f}GB). Skipping save.")
            return False
    
    try:
        torch.save(model_state_dict, filename)
        print(f"💾 Model saved successfully: {filename}")
        return True
    except OSError as e:
        if "No space left on device" in str(e):
            print(f"❌ Disk full! Could not save {filename}")
            cleanup_old_checkpoints()
            return False
        else:
            print(f"❌ Error saving {filename}: {e}")
            return False

def load_segmentation_model(path):
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

transform = transforms.ToTensor()

# Global frame cache to avoid repeated processing
frame_cache = {}
cache_lock = threading.Lock()

def segment_frame(model, frame, use_cache=True):
    """Optimized segmentation with caching"""
    frame_hash = hash(frame.tobytes()) if use_cache else None
    
    if use_cache and frame_hash in frame_cache:
        return frame_cache[frame_hash]
    
    resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    input_tensor = transform(resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    result = (mask, resized)
    
    if use_cache:
        with cache_lock:
            if len(frame_cache) > 10:  # Limit cache size
                frame_cache.clear()
            frame_cache[frame_hash] = result
    
    return result

def get_bbox(binary_mask):
    ys, xs = np.where(binary_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (min(xs), min(ys), max(xs), max(ys))

def get_centroid(binary_mask):
    ys, xs = np.where(binary_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))

def boxes_overlap(a, b):
    if a is None or b is None:
        return False
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

def frames_same(f1, f2, threshold=CRASH_THRESHOLD):
    """Optimized frame comparison"""
    if f1.shape != f2.shape:
        return False
    
    # Use faster numpy-based comparison instead of torchmetrics
    diff = np.mean((f1.astype(np.float32) - f2.astype(np.float32)) ** 2)
    similarity = 1.0 / (1.0 + diff / 1000.0)  # Normalize to 0-1 range
    return similarity >= threshold

def calculate_track_direction(mask):
    """Optimized track direction calculation"""
    car_mask = (mask == LABEL_CAR)
    track_mask = (mask == LABEL_TRACK)
    
    car_center = get_centroid(car_mask)
    if car_center is None:
        return 0
    
    # Sample fewer track pixels for speed
    track_ys, track_xs = np.where(track_mask)
    if len(track_xs) == 0:
        return 0
    
    # Sample every 10th pixel to reduce computation
    sample_size = min(len(track_xs), len(track_xs) // 10 + 1)
    indices = np.random.choice(len(track_xs), sample_size, replace=False)
    sampled_xs = track_xs[indices]
    
    car_x = car_center[0]
    left_weight = np.sum(sampled_xs < car_x)
    right_weight = np.sum(sampled_xs > car_x)
    
    if left_weight + right_weight == 0:
        return 0
    
    direction = (right_weight - left_weight) / (left_weight + right_weight)
    return np.clip(direction, -1, 1)

def calculate_car_speed(current_pos, prev_pos, dt=1/60):
    """Calculate car speed based on position change"""
    if current_pos is None or prev_pos is None:
        return 0
    
    distance = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
    speed = distance / dt  # pixels per second
    return speed

def calculate_progress(car_center, checkpoint_mask, finish_mask):
    """Calculate rough progress through the track"""
    if car_center is None:
        return 0
    
    # Simple progress calculation based on proximity to checkpoints/finish
    checkpoint_pixels = np.sum(checkpoint_mask)
    finish_pixels = np.sum(finish_mask)
    
    # This is a simplified progress calculation
    # In a real implementation, you might want to track lap progress more sophisticatedly
    progress = (checkpoint_pixels + finish_pixels * 2) / 100  # Normalize
    return progress

def calculate_action_complexity_penalty(action_idx, prev_action_idx):
    """Penalize erratic or contradictory actions"""
    if prev_action_idx is None:
        return 0
    
    # Count action changes
    changes = sum(1 for i in range(len(action_idx)) if action_idx[i] != prev_action_idx[i])
    
    # Penalize simultaneous opposite actions (left+right, forward+backward)
    penalty = 0
    if action_idx[0] == 1 and action_idx[1] == 1:  # left + right
        penalty -= 50
    if action_idx[2] == 1 and action_idx[3] == 1:  # forward + backward (unlikely but possible)
        penalty -= 100
    
    # Small penalty for too many simultaneous actions
    simultaneous_actions = sum(action_idx)
    if simultaneous_actions > 2:
        penalty -= 20
    
    # Penalty for too frequent action changes (jittery behavior)
    if changes > 2:
        penalty -= 10
    
    return penalty

def check_checkpoint_timeout(last_checkpoint_time, current_time):
    """Check if too much time has passed since last checkpoint"""
    if last_checkpoint_time is None:
        return False
    return (current_time - last_checkpoint_time) > CHECKPOINT_TIMEOUT

def detect_events(mask, prev_mask, f2, prev_positions, stuck_counter, checkpoint_counter, checkpoint_confirmed, finish_counter, finish_confirmed, last_checkpoint_time, current_time):
    """Optimized event detection"""
    global testingvar
    car_mask = (mask == LABEL_CAR)
    track_mask = (mask == LABEL_TRACK)
    checkpoint_mask = (mask == LABEL_CHECKPOINT)
    finish_mask = (mask == LABEL_FINISH)

    car_bbox = get_bbox(car_mask)
    car_center = get_centroid(car_mask)
    track_bbox = get_bbox(track_mask)

    events = []
    
    # Quick bounds check
    if car_bbox is None or not boxes_overlap(car_bbox, track_bbox):
        events.append("out_of_bounds")

    # Simplified checkpoint detection
    if np.any(checkpoint_mask & car_mask):
        checkpoint_counter += 1
        if checkpoint_counter >= CHECKPOINT_CONFIRM_FRAMES and not checkpoint_confirmed:
            events.append("checkpoint")
            checkpoint_confirmed = True
            last_checkpoint_time = current_time  # Update last checkpoint time
    else:
        checkpoint_counter = 0
        checkpoint_confirmed = False

    # Simplified finish line detection
    if np.any(finish_mask & car_mask):
        finish_counter += 1
        if finish_counter >= FINISH_CONFIRM_FRAMES and not finish_confirmed:
            events.append("finish")
            finish_confirmed = True
    else:
        finish_counter = 0
        finish_confirmed = False

    if check_checkpoint_timeout(last_checkpoint_time, current_time):
        events.append("checkpoint_timeout")

                        # After you calculate mask_next:
    if prev_mask is not None:
        similarity = np.mean(prev_mask == mask)
        if similarity < 0.97:
            testingvar = 0
        if similarity > 0.98:
            testingvar += 1
            if testingvar > 5:
                events.append("crash")
                testingvar = 0



    # Simplified stuck detection
    if car_center is not None:
        prev_positions.append(car_center)
        if len(prev_positions) > STUCK_THRESHOLD:
            # Only check every few frames to reduce computation
            if len(prev_positions) % 5 == 0:
                positions_list = list(prev_positions)
                distances = [np.linalg.norm(np.array(pos) - np.array(positions_list[0])) 
                           for pos in positions_list[-5:]]  # Check fewer positions
                if max(distances) < 8:
                    events.append("stuck")

    return events, stuck_counter + 1 if "stuck" in events else 0, checkpoint_counter, checkpoint_confirmed, finish_counter, finish_confirmed, last_checkpoint_time

def reward_from_events(events, episode_length, max_episode_length, track_direction, car_speed=0, progress_delta=0, consecutive_checkpoints=0, car_center=None, mask=None):
    reward = 0
    
    # Major completion rewards

    if car_center is not None and mask is not None:
        # Off-track penalty
        if is_car_off_track(car_center, mask):
            off_track_penalty = -50
            reward += off_track_penalty
            # print(f"🚫 Off track penalty: {off_track_penalty}")
        else:
            # Center-of-track reward
            distance_from_center = calculate_distance_from_track_center(car_center, mask)
            track_width = calculate_track_width(car_center, mask)
            
            if distance_from_center != float('inf') and track_width > 0:
                # Normalize distance (0 = center, 1 = edge)
                normalized_distance = distance_from_center / (track_width / 2)
                normalized_distance = min(normalized_distance, 1.5)  # Cap at 1.5 for extreme cases
                
                # Reward being close to center (exponential decay)
                center_reward = 50 * np.exp(-2 * normalized_distance)  # Max 10 points at center, ~1.4 at edge
                reward += center_reward
                
                # Additional penalty for being very close to track edge
                if normalized_distance > 0.8:
                    edge_penalty = -20 * (normalized_distance - 0.8)  # Up to -20 penalty
                    reward += edge_penalty

    if "finish" in events:
        # Massive reward for finishing, with time bonus
        time_efficiency_bonus = max(0, (max_episode_length - episode_length) / max_episode_length * 500)
        reward += 100000 + time_efficiency_bonus
        print(f"🏆 RACE COMPLETED! Time bonus: {time_efficiency_bonus:.1f}")
    

    elif "checkpoint" in events:
        # Base checkpoint reward with time bonus
        time_bonus = max(0, (max_episode_length - episode_length) / max_episode_length * 100)
        base_checkpoint_reward = 6000 + time_bonus
        
        # Consecutive checkpoint bonus (exponential growth)
        consecutive_bonus = min(consecutive_checkpoints * 200, 1000)  # Cap at 1000
        reward += base_checkpoint_reward + consecutive_bonus
        print(f"✅ Checkpoint! Base: {base_checkpoint_reward:.1f}, Consecutive bonus: {consecutive_bonus:.1f}")
    
    # Punishment system - scaled by survival time
    survival_factor = min(1.0, episode_length / 4000)  # 3 min
    
    if "crash" in events:
        # Less punishment for crashes that happen later (learned something)
        base_crash_penalty = -1500
        time_adjusted_penalty = base_crash_penalty * (1 - survival_factor * 0.9)  # Up to 70% reduction
        reward += time_adjusted_penalty
        print(f"💥 Crash penalty: {time_adjusted_penalty:.1f} (survival factor: {survival_factor:.2f})")
    
    if "stuck" in events:
        # Punishment for getting stuck, less severe if survived longer
        base_stuck_penalty = -1500
        time_adjusted_penalty = base_stuck_penalty * (1 - survival_factor * 0.9)  # Up to 50% reduction
        reward += time_adjusted_penalty
    
    if "out_of_bounds" in events:
        # Punishment for going off track, slightly reduced for longer survival
        base_oob_penalty = -2000
        time_adjusted_penalty = base_oob_penalty * (1 - survival_factor * 0.8)  # Up to 30% reduction
        reward += time_adjusted_penalty
    
    # Continuous rewards for good behavior
    
    # Time alive reward - encourages survival
    time_alive_reward = min(episode_length * 0.3, 50)  # Cap at 50 to prevent exploitation
    reward += time_alive_reward
    
    # Track direction reward - staying on racing line
    direction_reward = abs(track_direction) * 8
    reward += direction_reward
    
    # Speed reward - encourages forward movement
    if car_speed > 0:
        speed_reward = min(car_speed * 2, 20)  # Cap speed reward
        reward += speed_reward
    
    # Progress reward - reward for moving forward through the track
    if progress_delta > 0:
        progress_reward = progress_delta * 10
        reward += progress_reward
    elif progress_delta < 0:
        # Small penalty for going backwards
        reward += progress_delta * 5
    
    # Consistency bonus - small reward for each frame without crashing
    if not any(event in events for event in ["crash", "stuck", "out_of_bounds"]):
        consistency_bonus = 10
        reward += consistency_bonus
    if "checkpoint_timeout" in events:
        # Large penalty for not reaching checkpoint in time
        timeout_penalty = CHECKPOINT_TIMEOUT_PENALTY
        # Make penalty worse if it's been a long time since last checkpoint
        reward += timeout_penalty       
        print(f"⏱️ Checkpoint timeout penalty: {CHECKPOINT_TIMEOUT_PENALTY}")

    
    return reward

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape
        
        # Lighter CNN architecture for better performance
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        
        # Calculate the actual size by doing a forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_output = self.conv(dummy_input)
            linear_input_size = conv_output.view(1, -1).size(1)
        
        print(f"Input shape: {input_shape}")
        print(f"Calculated linear input size: {linear_input_size}")
        
        # Lighter FC layers
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)

def press_action(action_idx):
    for i in range(len(action_idx)):
        if action_idx[i] == 1:
            keyboard.press(ACTIONS[i])
        else:
            keyboard.release(ACTIONS[i])

def release_all_keys():
    for key in ACTIONS:
        keyboard.release(key)

def restart_track():
    print("🔄 Restarting track...")
    force_release_all_keys()  # Use the robust version
    time.sleep(0.2)  # Longer wait
    
    # More robust restart
    for _ in range(3):  # Try multiple times
        try:
            keyboard.press_and_release('Backspace')
            break
        except:
            time.sleep(0.1)
    
    time.sleep(3)
    force_release_all_keys()  # Ensure clean state

def calculate_distance_from_track_center(car_center, mask):
    """Calculate how far the car is from the center of the track"""
    if car_center is None:
        return float('inf')
    
    track_mask = (mask == LABEL_TRACK)
    track_ys, track_xs = np.where(track_mask)
    
    if len(track_xs) == 0:
        return float('inf')
    
    # Find track pixels in the same horizontal region as the car
    car_x, car_y = car_center
    horizontal_tolerance = 50  # pixels
    
    # Get track pixels near the car's Y position
    nearby_track_indices = np.where(np.abs(track_ys - car_y) <= horizontal_tolerance)[0]
    
    if len(nearby_track_indices) == 0:
        return float('inf')
    
    nearby_track_xs = track_xs[nearby_track_indices]
    nearby_track_ys = track_ys[nearby_track_indices]
    
    # Find the center of the track at this Y level
    track_center_x = np.mean(nearby_track_xs)
    track_center_y = np.mean(nearby_track_ys)
    
    # Calculate distance from car to track center
    distance = np.sqrt((car_x - track_center_x)**2 + (car_y - track_center_y)**2)
    return distance

def calculate_track_width(car_center, mask):
    """Calculate track width at car's position for normalization"""
    if car_center is None:
        return 100  # default width
    
    track_mask = (mask == LABEL_TRACK)
    track_ys, track_xs = np.where(track_mask)
    
    if len(track_xs) == 0:
        return 100
    
    car_x, car_y = car_center
    horizontal_tolerance = 20
    
    # Get track pixels at car's Y position
    nearby_indices = np.where(np.abs(track_ys - car_y) <= horizontal_tolerance)[0]
    
    if len(nearby_indices) == 0:
        return 100
    
    nearby_xs = track_xs[nearby_indices]
    track_width = np.max(nearby_xs) - np.min(nearby_xs)
    
    return max(track_width, 20)  # minimum width to avoid division issues

def is_car_off_track(car_center, mask):
    """Check if car is completely off the track"""
    if car_center is None:
        return True
    
    car_mask = (mask == LABEL_CAR)
    track_mask = (mask == LABEL_TRACK)
    
    # Check if any part of the car overlaps with track
    overlap = np.any(car_mask & track_mask)
    return not overlap

def preprocess_frame(frame):
    # """Ensure frame is properly preprocessed to match expected input shape"""
    # frame is already resized to INPUT_SIZE in segment_frame
    # Convert to tensor and ensure correct format
    if len(frame.shape) == 3 and frame.shape[2] == 3:  # HWC format
        tensor = transform(frame).float()  # This converts to CHW format
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    
    # print(f"Preprocessed frame shape: {tensor.shape}")  # Debug print
    return tensor.numpy()

def train_step(q_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states_v = torch.FloatTensor(states).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    dones_v = torch.FloatTensor(dones).to(device)

    q_values = q_net(states_v)
    
    with torch.no_grad():
        next_q_values = target_net(next_states_v)
        next_q_max = next_q_values.max(1)[0]
        expected_q_values = rewards_v + (GAMMA * next_q_max * (1 - dones_v))

    state_action_values = (q_values * actions_v).sum(1)
    loss = F.mse_loss(state_action_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    optimizer.step()

def main():
    # Check initial disk space
    free_space = check_disk_space()
    print(f"💾 Available disk space: {free_space:.2f}GB")
    reward_history = []

    if free_space < MIN_FREE_SPACE_GB * 2:  # Need at least 2GB to start safely
        print("⚠️ Warning: Low disk space! Consider freeing up space before training.")
        cleanup_old_checkpoints()
    
    print("🔧 Loading segmentation model...")
    seg_model = load_segmentation_model(MODEL_PATH)

    q_net = QNetwork((3, INPUT_SIZE[0], INPUT_SIZE[1]), NUM_ACTIONS).to(device)  # Note: swapped order
    target_net = QNetwork((3, INPUT_SIZE[0], INPUT_SIZE[1]), NUM_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)


    epsilon = EPSILON_START
    epsilon_decay_step = 0.00005
    step_idx = 0

    monitor = find_trackmania_window()
    print(f"🎮 Capturing Trackmania window at: {monitor}")
    testingvar = 0
    total_episodes = 1000
    prev_mask = None
    
    with mss.mss() as sct:
        try:
            for episode in range(total_episodes):
                events = []
                episode_reward = 0
                episode_length = 0
                episode_start_time = time.time()
                prev_positions = deque(maxlen=STUCK_THRESHOLD)
                stuck_counter = 0
                checkpoint_counter = 0
                checkpoint_confirmed = False
                finish_counter = 0
                finish_confirmed = False
                frame_count = 0
                consecutive_checkpoints = 0
                max_consecutive_checkpoints = 0
                print(f"\n🚗 Starting Episode {episode + 1}/{total_episodes} (ε={epsilon:.3f})")

                prev_frame = None
                prev_car_pos = None
                prev_action_idx = None
                prev_progress = 0
                last_checkpoint_time = time.time()  # Initialize checkpoint timer at episode start

                while True:
                    current_time = time.time()
                    if current_time - episode_start_time > EPISODE_TIMEOUT:
                        print("⏰ Episode timeout - restarting track")
                        episode_reward -= 1000
                        restart_track()
                        break
                    if episode_length % 100 == 0:
                        force_release_all_keys()
                        time.sleep(0.05)
                    # Skip frames for performance
                    frame_count += 1

                    try:
                        sct_img = sct.grab(monitor)
                        frame = np.array(sct_img)[..., :3]
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        try:
                            mask, resized_frame = segment_frame(seg_model, frame)
                            if resized_frame is None or resized_frame.shape != (INPUT_SIZE[1], INPUT_SIZE[0], 3):
                                raise ValueError("Invalid resized frame")
                            
                            state = preprocess_frame(resized_frame)
                            if state is None or np.isnan(state).any():
                                raise ValueError("Invalid state after preprocessing")
                        except Exception as e:
                            continue

                        track_direction = calculate_track_direction(mask)
                        
                        # Calculate additional metrics for reward system
                        car_mask = (mask == LABEL_CAR)
                        car_center = get_centroid(car_mask)
                        car_speed = calculate_car_speed(car_center, prev_car_pos) if prev_car_pos else 0
                        
                        # Calculate progress (simplified)
                        checkpoint_mask = (mask == LABEL_CHECKPOINT)
                        finish_mask = (mask == LABEL_FINISH)
                        current_progress = calculate_progress(car_center, checkpoint_mask, finish_mask)
                        progress_delta = current_progress - prev_progress
                        
                        # Action selection
                        if random.random() < epsilon:
                            action_idx = []
                            for i, weight in enumerate(ACTION_WEIGHTS):
                                if i == 0:
                                    adjusted_weight = weight * (1.5 if track_direction > -0.3 else 0.5)
                                elif i == 1:
                                    adjusted_weight = weight * (1.5 if track_direction < 0.3 else 0.5)
                                else:
                                    adjusted_weight = weight
                                
                                action_idx.append(1 if random.random() <= adjusted_weight else 0)
                        else:
                            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                            with torch.no_grad():
                                q_values = q_net(state_t)
                                probs = torch.sigmoid(q_values.squeeze())
                                action_idx = [1 if p > 0.5 else 0 for p in probs.tolist()]
                        
                        # Calculate action complexity penalty
                        action_penalty = calculate_action_complexity_penalty(action_idx, prev_action_idx)

                        press_action(action_idx)

                        # Simplified next frame processing
                        time.sleep(1 / 60)
                        sct_img_next = sct.grab(monitor)
                        frame_next = np.array(sct_img_next)[..., :3]
                        frame_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2RGB)
                        try:
                            mask_next, resized_frame_next = segment_frame(seg_model, frame_next)
                            if resized_frame_next is None or resized_frame_next.shape != (INPUT_SIZE[1], INPUT_SIZE[0], 3):
                                raise ValueError("Invalid resized frame (next)")

                            next_state = preprocess_frame(resized_frame_next)
                            if next_state is None or np.isnan(next_state).any():
                                raise ValueError("Invalid next_state after preprocessing")
                        except Exception as e:
                            continue
                        
                        done = False


                        # Event detection (less frequent)
                        
                        if prev_frame is not None and episode_length % 3 == 0:
                            events, stuck_counter, checkpoint_counter, checkpoint_confirmed, finish_counter, finish_confirmed, last_checkpoint_time = detect_events(
                                mask_next, prev_mask, frame_next, prev_positions, stuck_counter, 
                                checkpoint_counter, checkpoint_confirmed, finish_counter, finish_confirmed, last_checkpoint_time, current_time)

                        reward = reward_from_events(events, episode_length, EPISODE_TIMEOUT * 60, track_direction, car_speed, progress_delta, consecutive_checkpoints, car_center, mask_next)
                        reward_history.append(reward)

                        prev_mask = mask_next.copy()


                        
                        # Add action complexity penalty
                        reward += action_penalty
                        
                        # Progress display (less frequent)
                        if finish_counter > 0 and not finish_confirmed and episode_length % 30 == 0:
                            progress = (finish_counter / FINISH_CONFIRM_FRAMES) * 100
                            print(f"🏁 Finish line: {progress:.1f}%")
                        
                        if "finish" in events:
                            print(f"🏁 Track completed! Reward: {episode_reward:.2f}")
                            done = True
                        if "crash" in events or "stuck" in events or stuck_counter > 10:
                            print(f"💥 {'Crash' if 'crash' in events else 'Stuck'} - restarting")
                            restart_track()
                            done = True
                        if "checkpoint" in events:
                            consecutive_checkpoints += 1
                            max_consecutive_checkpoints = max(max_consecutive_checkpoints, consecutive_checkpoints)
                            last_checkpoint_time = time.time()  # Reset timer when checkpoint is reached
                            print(f"✅ Checkpoint! Reward: {reward:.1f}")

                        episode_reward += reward
                        
                        # Training (less frequent)
                        if prev_frame is not None and step_idx % TRAIN_FREQUENCY == 0:
                            resized_prev_frame = cv2.resize(prev_frame, INPUT_SIZE)
                            prev_state = preprocess_frame(resized_prev_frame)
                            if events:
                                print(f"📍 Events detected: {events}")

                            # replay_buffer.push(prev_state, action_idx, reward, next_state, done, events)
                            replay_buffer.push(prev_state, action_idx, reward, next_state, done)

                            train_step(q_net, target_net, optimizer, replay_buffer)

                            # train_step_prioritized(q_net, target_net, optimizer, replay_buffer)



                        if step_idx % TARGET_UPDATE_FREQ == 0:
                            target_net.load_state_dict(q_net.state_dict())
                            print(f"🔄 Target network updated at step {step_idx}")

                        if epsilon > EPSILON_END:
                            epsilon -= epsilon_decay_step

                        step_idx += 1
                        episode_length += 1
                        prev_frame = frame_next.copy()
                        prev_car_pos = car_center
                        prev_action_idx = action_idx.copy()
                        prev_progress = current_progress

                        if done:
                            last_checkpoint_time = time.time()  # Reset checkpoint timer for new episode
                            break

                    except Exception as e:
                        continue

                print(f"📊 Episode {episode + 1} - Reward: {episode_reward:.2f}, Steps: {episode_length}, ε: {epsilon:.3f}, Avg Reward/Step: {episode_reward/max(1, episode_length):.2f}")
                
                # IMPROVED CHECKPOINT SAVING with disk space management
                if (episode + 1) % CHECKPOINT_SAVE_FREQUENCY == 0:
                    checkpoint_filename = f"trackmania_dqn_checkpoint_{episode + 1}.pth"
                    if safe_save_model(q_net.state_dict(), checkpoint_filename):
                        # Clean up old checkpoints after successful save
                        cleanup_old_checkpoints()
                    
                    # Clear cache periodically
                    with cache_lock:
                        frame_cache.clear()

            print("🎉 Training completed!")
            
            # Final save with safety check
            if not safe_save_model(q_net.state_dict(), "trackmania_dqn_final_optimized.pth"):
                print("⚠️ Could not save final model due to disk space issues")

        except KeyboardInterrupt:
            print("🛑 Training interrupted by user.")
        finally:
            release_all_keys()
            # Try to save final model even if interrupted
            plt.figure(figsize=(10, 4))
            plt.plot(reward_history, label="Reward per frame")
            plt.xlabel("Timestep / Frame")
            plt.ylabel("Reward")
            plt.title("Reward over Time")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
            safe_save_model(q_net.state_dict(), "trackmania_dqn_interrupted.pth")

if __name__ == "__main__":
    main()