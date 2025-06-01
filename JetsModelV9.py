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

from queue import Queue
import threading

import os
os.environ["TMPDIR"] = "F:/temp"
os.environ["TEMP"] = "F:/temp"
os.environ["TMP"] = "F:/temp"

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
GAMMA = 0.95
LEARNING_RATE = 1e-4
REPLAY_BUFFER_CAPACITY = 200000    # Reduced buffer size
TARGET_UPDATE_FREQ = 100  # Less frequent updates
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000 #change to 50000

# Performance optimizations
EPISODE_TIMEOUT = 300
CRASH_THRESHOLD = 0.87
STUCK_THRESHOLD = 10
CHECKPOINT_CONFIRM_FRAMES = 2
FINISH_CONFIRM_FRAMES = 30  # 5 seconds at 60 FPS
VELOCITY_THRESHOLD = 3
FRAME_SKIP = 4  # Process every 2nd frame to reduce load
TRAIN_FREQUENCY = 1  # Train every 4 steps instead of every step
CHECKPOINT_TIMEOUT = 100  # seconds without checkpoint before penalty
CHECKPOINT_TIMEOUT_PENALTY = -500  # Large penalty for not reaching checkpoint in time


# DISK SPACE MANAGEMENT
CHECKPOINT_SAVE_FREQUENCY = 50  # Save less frequently (every 50 episodes instead of 25)
MAX_CHECKPOINTS_TO_KEEP = 3     # Only keep the 3 most recent checkpoints
MIN_FREE_SPACE_GB = 1.0         # Minimum free space to maintain (in GB)

testingvar = 0
checkpoint_start_time = None
last_checkpoint_reach_time = None
# Thread control variables
action_queue = Queue()
control_thread = None
stop_control_thread = False
current_action = [0, 0, 0, 0]  # Current action state
action_lock = threading.Lock()

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization is used
        self.beta = beta    # Importance sampling correction
        self.beta_increment = beta_increment
        
        # Use deque for efficient operations
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done, events=None):
        """Add experience with maximum priority for new experiences"""
        experience = (state, action, reward, next_state, done, events)
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size):
        """Sample batch with priorities"""
        if len(self.buffer) < batch_size:
            return None
            
        # Convert to numpy arrays for efficient computation
        priorities = np.array(self.priorities)
        
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones, events_list = zip(*experiences)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Increment beta for annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.uint8), 
                weights, indices)
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha  # Small epsilon to avoid zero priority
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)
    

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
    """Dedicated thread for handling WASD controls"""
    global stop_control_thread, current_action
    
    while not stop_control_thread:
        try:
            # Check for new actions (non-blocking)
            try:
                new_action = action_queue.get_nowait()
                with action_lock:
                    current_action = new_action
                action_queue.task_done()
            except:
                pass  # No new action, continue with current
            
            # Execute current action
            with action_lock:
                action_to_execute = current_action.copy()
            
            # Press keys based on current action
            for i, should_press in enumerate(action_to_execute):
                if should_press:
                    keyboard.press(ACTIONS[i])
                else:
                    keyboard.release(ACTIONS[i])
            
            time.sleep(0.016)  # ~60 FPS for smooth control
            
        except Exception as e:
            print(f"Control thread error: {e}")
            time.sleep(0.1)
    
    # Clean up - release all keys when thread stops
    force_release_all_keys()

def force_release_all_keys():
    """Force release all keys using Windows API"""
    for key in ACTIONS:
        # Convert to virtual key code
        vk_code = ord(key.upper())
        # Force key up using Windows API
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.01)



def train_step_prioritized(q_net, target_net, optimizer, replay_buffer):
    """Enhanced training with prioritized experience replay"""
    if len(replay_buffer) < BATCH_SIZE:
        return None
    
    # Sample from prioritized buffer
    sample_result = replay_buffer.sample(BATCH_SIZE)
    if sample_result is None:
        return None
    
    states, actions, rewards, next_states, dones, weights, indices = sample_result
    
    states_v = torch.FloatTensor(states).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    dones_v = torch.FloatTensor(dones).to(device)
    weights_v = torch.FloatTensor(weights).to(device)
    
    # Current Q values
    q_values = q_net(states_v)
    state_action_values = (q_values * actions_v).sum(1)
    
    # Next Q values using Double DQN
    with torch.no_grad():
        # Use main network to select actions
        next_q_main = q_net(next_states_v)
        next_actions = next_q_main.max(1)[1].unsqueeze(1)
        
        # Use target network to evaluate actions
        next_q_target = target_net(next_states_v)
        next_q_max = next_q_target.gather(1, next_actions).squeeze()
        
        expected_q_values = rewards_v + (GAMMA * next_q_max * (1 - dones_v))
    
    # Calculate TD errors for priority updates
    td_errors = (expected_q_values - state_action_values).detach().cpu().numpy()
    
    # Weighted loss using importance sampling
    loss = (weights_v * F.smooth_l1_loss(state_action_values, expected_q_values, reduction='none')).mean()
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    optimizer.step()
    
    # Update priorities
    replay_buffer.update_priorities(indices, td_errors)
    
    return loss.item()


def calculate_experience_bonus(events, replay_buffer):
    """Give bonus for rare/important experiences"""
    bonus = 0
    
    # Count how often similar events occurred recently
    if len(replay_buffer) > 1000:
        recent_experiences = list(replay_buffer.buffer)[-1000:]  # Last 1000 experiences
        
        checkpoint_count = sum(1 for exp in recent_experiences if exp[5] and "checkpoint" in exp[5])
        finish_count = sum(1 for exp in recent_experiences if exp[5] and "finish" in exp[5])
        
        # Bonus for rare positive events
        if "checkpoint" in events:
            rarity_bonus = max(0, 100 - checkpoint_count * 10)  # Less bonus if checkpoints are common
            bonus += rarity_bonus
            
        if "finish" in events:
            rarity_bonus = max(0, 500 - finish_count * 50)  # Big bonus for rare finishes
            bonus += rarity_bonus
    
    return bonus


class MultiStepBuffer:
    def __init__(self, n_steps=3, gamma=0.99):
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=n_steps)
    
    def add(self, state, action, reward, next_state, done):
        """Add transition to n-step buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
        if len(self.buffer) == self.n_steps:
            # Calculate n-step return
            n_step_return = 0
            for i, (_, _, r, _, d) in enumerate(self.buffer):
                n_step_return += (self.gamma ** i) * r
                if d:  # Episode ended
                    break
            
            # Return n-step transition
            first_state, first_action = self.buffer[0][:2]
            last_next_state, last_done = self.buffer[-1][3:]
            
            return (first_state, first_action, n_step_return, last_next_state, last_done)
        
        return None
    
    def clear(self):
        self.buffer.clear()








def initialize_enhanced_replay_system():
    """Initialize the enhanced replay system"""
    # Replace regular ReplayBuffer with PrioritizedReplayBuffer
    replay_buffer = PrioritizedReplayBuffer(
        capacity=REPLAY_BUFFER_CAPACITY,
        alpha=0.6,      # Prioritization strength
        beta=0.4,       # Importance sampling correction
        beta_increment=0.001  # Beta annealing rate
    )
    
    # Optional: Add multi-step learning
    multi_step_buffer = MultiStepBuffer(n_steps=3, gamma=GAMMA)
    
    return replay_buffer, multi_step_buffer

# Modified experience storage in your main loop:
def store_experience_enhanced(replay_buffer, multi_step_buffer, prev_state, action_idx, reward, next_state, done, events):
    """Enhanced experience storage with multi-step learning"""
    
    # Add experience bonus for rare events
    experience_bonus = calculate_experience_bonus(events, replay_buffer)
    enhanced_reward = reward + experience_bonus
    
    # Store in multi-step buffer
    n_step_transition = multi_step_buffer.add(prev_state, action_idx, enhanced_reward, next_state, done)
    
    # If we have a complete n-step transition, add to replay buffer
    if n_step_transition is not None:
        state, action, n_step_return, final_next_state, final_done = n_step_transition
        replay_buffer.push(state, action, n_step_return, final_next_state, final_done, events)
    
    # Always store single-step transition as well (for diversity)
    replay_buffer.push(prev_state, action_idx, enhanced_reward, next_state, done, events)
    
    # Clear multi-step buffer if episode ended
    if done:
        multi_step_buffer.clear()



class AdaptiveTraining:
    def __init__(self, initial_freq=4, min_freq=1, max_freq=8):
        self.initial_freq = initial_freq
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.current_freq = initial_freq
        self.loss_history = deque(maxlen=100)
    
    def should_train(self, step_idx):
        """Decide whether to train based on adaptive frequency"""
        return step_idx % self.current_freq == 0
    
    def update_frequency(self, loss):
        """Adjust training frequency based on loss"""
        if loss is not None:
            self.loss_history.append(loss)
            
            if len(self.loss_history) >= 50:
                recent_loss = np.mean(list(self.loss_history)[-20:])
                older_loss = np.mean(list(self.loss_history)[-50:-20])
                
                # If loss is improving, train less frequently
                if recent_loss < older_loss * 0.9:
                    self.current_freq = min(self.max_freq, self.current_freq + 1)
                # If loss is getting worse, train more frequently
                elif recent_loss > older_loss * 1.1:
                    self.current_freq = max(self.min_freq, self.current_freq - 1)








device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CODEITERATION7")
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
    """Calculate car speed based on position change - FIXED VERSION"""
    if current_pos is None or prev_pos is None:
        return 0
    
    # Convert to numpy arrays for consistent calculation
    current_pos = np.array(current_pos, dtype=np.float32)
    prev_pos = np.array(prev_pos, dtype=np.float32)
    
    # Calculate Euclidean distance between positions
    distance = np.linalg.norm(current_pos - prev_pos)
    
    # Calculate speed (pixels per second)
    speed = distance / dt
    
    # Clamp speed to reasonable values to avoid outliers
    speed = np.clip(speed, 0, 500)  # Max reasonable speed in pixels/second
    
    return float(speed)


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

def detect_track_coverage_crash(mask):

    track_mask = (mask == LABEL_TRACK)
    track_pixels = np.sum(track_mask)
    
    # Calculate total pixels in the frame
    total_pixels = mask.shape[0] * mask.shape[1]
    
    # Calculate track coverage ratio
    track_coverage_ratio = track_pixels / total_pixels
    
    # Return True if coverage is below threshold (crash condition)
    is_crash = track_coverage_ratio < 0.20
    
    if is_crash:
        print(f"Track coverage crash detected! Coverage: {track_coverage_ratio:.3f} ({track_coverage_ratio*100:.1f}%)")
    
    return is_crash


def simple_checkpoint_detection(mask, prev_checkpoint_pixels=0):
    """Simple checkpoint detection based on screen coverage"""
    checkpoint_mask = (mask == LABEL_CHECKPOINT)
    current_checkpoint_pixels = np.sum(checkpoint_mask)
    
    # Calculate what percentage of screen the checkpoint takes up
    total_pixels = mask.shape[0] * mask.shape[1]
    checkpoint_ratio = current_checkpoint_pixels / total_pixels
    
    threshold = 0.07  # 1.5% of screen
    
    if checkpoint_ratio > threshold:
        print(f"Checkpoint detected! Screen coverage: {checkpoint_ratio:.3f} ({checkpoint_ratio*100:.1f}%)")
        return True
    
    return False

def enhanced_checkpoint_detection(mask, prev_checkpoint_coverage=0, checkpoint_counter=0, checkpoint_confirmed=False):
    """Enhanced checkpoint detection with progressive rewards"""
    checkpoint_mask = (mask == LABEL_CHECKPOINT)
    car_mask = (mask == LABEL_CAR)
    
    current_checkpoint_pixels = np.sum(checkpoint_mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    checkpoint_coverage = current_checkpoint_pixels / total_pixels
    
    events = []
    

    if checkpoint_coverage > 0.001:  # Checkpoint visible
        events.append("checkpoint_visible")
    
    if checkpoint_coverage > 0.03:   # Getting close
        events.append("checkpoint_approaching")
    
    if checkpoint_coverage > 0.07:   # Very close - your original threshold
        checkpoint_counter += 1
        if checkpoint_counter >= CHECKPOINT_CONFIRM_FRAMES and not checkpoint_confirmed:
            events.append("checkpoint_reached")
            checkpoint_confirmed = True
            print(f"Checkpoint reached! Coverage: {checkpoint_coverage:.3f} ({checkpoint_coverage*100:.1f}%)")
    else:
        checkpoint_counter = 0
        checkpoint_confirmed = False

    
    return events, checkpoint_coverage, checkpoint_counter, checkpoint_confirmed

def detect_events(mask, prev_mask, f2, prev_positions, stuck_counter, checkpoint_counter, checkpoint_confirmed, finish_counter, finish_confirmed, last_checkpoint_time, current_time, prev_checkpoint_coverage=0):
    """Optimized event detection - MODIFIED to return current_time"""
    global testingvar, last_checkpoint_reach_time
    car_mask = (mask == LABEL_CAR)
    track_mask = (mask == LABEL_TRACK)
    checkpoint_mask = (mask == LABEL_CHECKPOINT)
    finish_mask = (mask == LABEL_FINISH)
    last_checkpoint_reach_time = last_checkpoint_time
    car_bbox = get_bbox(car_mask)
    car_center = get_centroid(car_mask)
    track_bbox = get_bbox(track_mask)

    events = []
    
    # Quick bounds check
    if car_bbox is None or not boxes_overlap(car_bbox, track_bbox):
        events.append("out_of_bounds")

    if detect_track_coverage_crash(mask):
        events.append("crash")

    checkpoint_events, checkpoint_coverage, checkpoint_counter, checkpoint_confirmed = enhanced_checkpoint_detection(
        mask, prev_checkpoint_coverage, checkpoint_counter, checkpoint_confirmed
    )
    events.extend(checkpoint_events)

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

    # Crash detection based on similarity
    if current_time - last_checkpoint_time > 5:
        if prev_mask is not None:
            similarity = np.mean(prev_mask == mask)
            if similarity < 0.98:
                testingvar = 0
            if similarity > 0.985:
                testingvar += 1
                if testingvar > 7:
                    events.append("crash")
                    print("similarity crash")
                    testingvar = 0

    if car_center is not None:
        prev_positions.append(car_center)
        if len(prev_positions) > STUCK_THRESHOLD:
            if len(prev_positions) % 5 == 0:
                positions_list = list(prev_positions)
                distances = [np.linalg.norm(np.array(pos) - np.array(positions_list[0])) 
                           for pos in positions_list[-5:]]
                if max(distances) < 8:
                    events.append("stuck")

    return events, stuck_counter + 1 if "stuck" in events else 0, checkpoint_counter, checkpoint_confirmed, finish_counter, finish_confirmed, last_checkpoint_time, checkpoint_coverage


def reward_from_events(events, episode_length, max_episode_length, track_direction, checkpoint_coverage=0, prev_checkpoint_coverage=0, car_speed=0, progress_delta=0, consecutive_checkpoints=0, car_center=None, mask=None, current_time=None):
    global checkpoint_start_time, last_checkpoint_reach_time
    checkpointhresholdtime = 1
    reward = 0
    
    if "finish" in events:
        reward += 1000
        print(f"🏆 RACE COMPLETED!")
    
    elif "checkpoint" in events:
        base_checkpoint_reward = 75
        consecutive_bonus = consecutive_checkpoints * 75
        
        # SPEED BONUS BASED ON TIME BETWEEN CHECKPOINTS
        time_bonus = 0
        if last_checkpoint_reach_time is not None and current_time is not None:
            time_since_last_checkpoint = current_time - last_checkpoint_reach_time
            print(time_since_last_checkpoint)
            # Award bonus based on speed (less time = more bonus)
            if time_since_last_checkpoint < checkpointhresholdtime:
                time_bonus = -5
                print("tried cheating")
            elif time_since_last_checkpoint < 3.0:  # Very fast (under 5 seconds)
                time_bonus = 100
                print(f"🚀 LIGHTNING FAST! Time: {time_since_last_checkpoint:.1f}s, Bonus: {time_bonus}")
            elif time_since_last_checkpoint < 5.0:  # Fast (under 8 seconds)
                time_bonus = 75
                print(f"⚡ VERY FAST! Time: {time_since_last_checkpoint:.1f}s, Bonus: {time_bonus}")
            elif time_since_last_checkpoint < 10.0:  # Good (under 12 seconds)
                time_bonus = 50
                print(f"🏃 FAST! Time: {time_since_last_checkpoint:.1f}s, Bonus: {time_bonus}")
            elif time_since_last_checkpoint < 20.0:  # OK (under 20 seconds)
                time_bonus = 25
                print(f"👍 GOOD TIME! Time: {time_since_last_checkpoint:.1f}s, Bonus: {time_bonus}")
            else:  # Slow (over 20 seconds)
                time_bonus = 0
                print(f"🐌 Slow checkpoint... Time: {time_since_last_checkpoint:.1f}s, Bonus: {time_bonus}")
        
        # Update the last checkpoint time
        if time_bonus != -2:
            total_checkpoint_reward = base_checkpoint_reward + consecutive_bonus + time_bonus
        else:
            total_checkpoint_reward = time_bonus
        reward += total_checkpoint_reward
        print(f"✅ Checkpoint! Base: {base_checkpoint_reward}, Consecutive: {consecutive_bonus}, Speed: {time_bonus}, Total: {total_checkpoint_reward}")

    if "checkpoint_visible" in events:
        coverage_reward, _ = calculate_checkpoint_coverage_reward(mask, prev_checkpoint_coverage)
        reward += coverage_reward
        if coverage_reward > 10:
            print(f"🎯 Checkpoint coverage reward: {coverage_reward:.1f} (coverage: {checkpoint_coverage*100:.1f}%)")
    
    if "checkpoint_approaching" in events:
        reward += 15
    
    # Penalties (kept the same)
    if "crash" in events:
        reward -= 50
    
    if "stuck" in events:
        reward -= 30
    
    if "out_of_bounds" in events:
        reward -= 40
    
    if "checkpoint_timeout" in events:
        reward -= 5
    
    # Continuous rewards (kept small as you had them)
    # time_alive_reward = 0.1
    # reward += time_alive_reward
    
    direction_reward = 10-(track_direction*10)
    reward += direction_reward
    
    # Fixed speed reward using the corrected car speed
    # if car_speed > 0:
    #     speed_reward = min(car_speed * 0.02, 5)  # Scale down the speed reward appropriately
    #     reward += speed_reward
    
    # Consistency bonus
    # if not any(event in events for event in ["crash", "stuck", "out_of_bounds"]):
    #     consistency_bonus = 2
    #     reward += consistency_bonus

        
    return reward

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape
        
        # Better CNN architecture with batch normalization
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
        
        # Better FC layers with dropout
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
    """Queue action for the dedicated control thread"""
    global action_queue
    
    try:
        # Clear any pending actions and add the new one
        while not action_queue.empty():
            try:
                action_queue.get_nowait()
                action_queue.task_done()
            except:
                break
        
        action_queue.put(action_idx.copy())
    except Exception as e:
        print(f"Error queuing action: {e}")

def release_all_keys():
    for key in ACTIONS:
        keyboard.release(key)

def restart_track():
    print("🔄 Restarting track...")
    
    # Clear action queue
    while not action_queue.empty():
        try:
            action_queue.get_nowait()
            action_queue.task_done()
        except:
            break
    
    # Set action to no movement
    with action_lock:
        current_action = [0, 0, 0, 0]
    
    force_release_all_keys()
    time.sleep(0.2)
    
    for _ in range(3):
        try:
            keyboard.press_and_release('Backspace')
            break
        except:
            time.sleep(0.1)
    
    time.sleep(2)
    force_release_all_keys()

def select_action(q_net, state, epsilon, track_direction, device='cuda'):
    # ε-greedy with structured + stochastic exploration
    if random.random() < epsilon:
        action_idx = [0, 0, 0, 0]

        # 🔁 Occasionally go fully random (10% of the time)
        if random.random() < 0.1:
            return [random.randint(0, 1) for _ in range(4)]

        # 🎯 Scaled Directional Biasing
        # The further off-center, the more likely it is to turn
        turn_prob = min(max(abs(track_direction), 0.1), 1.0)  # Clamp between 0.1 and 1.0

        if track_direction > 0:  # On right side → turn left
            if random.random() < turn_prob:
                action_idx[1] = 1  # Turn Left
        elif track_direction < 0:  # On left side → turn right
            if random.random() < turn_prob:
                action_idx[0] = 1  # Turn Right

        # print(f"Track dir: {track_direction:.2f}, Turn prob: {turn_prob:.2f}")

        # 🚗 Forward/Backward Bias
        if random.random() < 0.9:
            action_idx[2] = 1  # Forward
        elif random.random() < 0.1:
            action_idx[3] = 1  # Backward

        # 🛑 Occasionally no movement
        if random.random() < 0.2:
            return [0, 0, 0, 0]
        

        return action_idx

    else:
        # 💡 Policy-driven action using Q-values
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_net(state_t)
            action_probs = torch.sigmoid(q_values.squeeze())

            # 🎯 Adaptive thresholds (can tweak based on performance)
            action_idx = [
                int(action_probs[0].item() > 0.5),  # Right
                int(action_probs[1].item() > 0.5),  # Left
                int(action_probs[2].item() > 0.4),  # Forward
                int(action_probs[3].item() > 0.6),  # Backward
            ]

        return action_idx


    
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

def calculate_checkpoint_coverage_reward(mask, prev_checkpoint_coverage=0):
    """Calculate reward based on checkpoint screen coverage"""
    d=3
    checkpoint_mask = (mask == LABEL_CHECKPOINT)
    checkpoint_pixels = np.sum(checkpoint_mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    current_coverage = checkpoint_pixels / total_pixels
    
    # Progressive reward based on coverage
    coverage_reward = 0
    
    if current_coverage > 0.001:  # Very small checkpoint visible
        coverage_reward += 5/d
    if current_coverage > 0.01:   # 1% of screen
        coverage_reward += 10/d
    if current_coverage > 0.03:   # 3% of screen
        coverage_reward += 20/d
    if current_coverage > 0.05:   # 5% of screen - getting close
        coverage_reward += 35/d
    if current_coverage > 0.07:   # 7% of screen - very close
        coverage_reward += 50/d
    if current_coverage > 0.10:   # 10% of screen - extremely close
        coverage_reward += 75/d
    
    # Bonus for increasing coverage (approaching checkpoint)
    coverage_increase = current_coverage - prev_checkpoint_coverage
    # if coverage_increase > 0:
    #     approach_bonus = coverage_increase * 1000  # Scale the increase
    #     coverage_reward += approach_bonus
    
    return coverage_reward, current_coverage

def train_step(q_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE * 4:  # Wait for more experience
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states_v = torch.FloatTensor(states).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    dones_v = torch.FloatTensor(dones).to(device)

    # Double DQN improvement
    q_values = q_net(states_v)
    
    with torch.no_grad():
        # Use main network to select actions
        next_q_main = q_net(next_states_v)
        next_actions = next_q_main.max(1)[1].unsqueeze(1)
        
        # Use target network to evaluate actions
        next_q_target = target_net(next_states_v)
        next_q_max = next_q_target.gather(1, next_actions).squeeze()
        
        expected_q_values = rewards_v + (GAMMA * next_q_max * (1 - dones_v))

    state_action_values = (q_values * actions_v).sum(1)
    
    # Huber loss instead of MSE - more stable
    loss = F.smooth_l1_loss(state_action_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 0.5)
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
    start_control_thread()
    q_net = QNetwork((3, INPUT_SIZE[0], INPUT_SIZE[1]), NUM_ACTIONS).to(device)  # Note: swapped order
    target_net = QNetwork((3, INPUT_SIZE[0], INPUT_SIZE[1]), NUM_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    replay_buffer, multi_step_buffer = initialize_enhanced_replay_system()
    adaptive_trainer = AdaptiveTraining()

    epsilon = EPSILON_START
    epsilon_decay_step = 0.00005
    step_idx = 0

    monitor = find_trackmania_window()
    print(f"🎮 Capturing Trackmania window at: {monitor}")
    testingvar = 0
    total_episodes = 300
    prev_mask = None
    
    with mss.mss() as sct:
        try:
            for episode in range(total_episodes):
                track_direction_timeout = 0
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
                prev_checkpoint_coverage = 0
                current_checkpoint_coverage = 0
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
                        time.sleep(0.005)
                    # Skip frames for performance
                    frame_count += 1
                    # print(time.time()-last_checkpoint_time)
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
                        action_idx = select_action(q_net, state, epsilon, track_direction)

                        press_action(action_idx)
                        
                        # Calculate action complexity penalty


                        # Simplified next frame processing
                        time.sleep(0.005)  # Slower to see action effect
                        sct_img_next = sct.grab(monitor)
                        frame_next = np.array(sct_img_next)[..., :3]
                        frame_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2RGB)

                        try:
                            mask_next, resized_frame_next = segment_frame(seg_model, frame_next)
                            next_state = preprocess_frame(resized_frame_next)
                        except Exception as e:
                            continue
                        
                        done = False


                        # Event detection (less frequent)

                        if prev_frame is not None:
                            # Calculate reward BEFORE adding to buffer
                            events, stuck_counter, checkpoint_counter, checkpoint_confirmed, finish_counter, finish_confirmed, last_checkpoint_time, current_checkpoint_coverage = detect_events(
                                mask_next, prev_mask, frame_next, prev_positions, stuck_counter, 
                                checkpoint_counter, checkpoint_confirmed, finish_counter, finish_confirmed, last_checkpoint_time, current_time, prev_checkpoint_coverage)
                            # if track_direction > 0.97 or track_direction < -0.97:
                            #     events.append("crash")
                            # if track_direction < 0.01 and track_direction > -0.01:
                            #     track_direction_timeout +=1
                            #     if track_direction_timeout > 5:
                            #         events.append("crash")
                            # else:
                            #     track_direction_timeout = 0

                            checkpoint_detected = simple_checkpoint_detection(mask_next)

                            if checkpoint_detected:
                                checkpoint_counter += 1
                                if checkpoint_counter >= CHECKPOINT_CONFIRM_FRAMES and not checkpoint_confirmed:
                                    events.append("checkpoint")
                                    checkpoint_confirmed = True
                                    last_checkpoint_time = current_time
                                    
                                    print("checkpoint reached")
                            else:
                                checkpoint_counter = 0
                                checkpoint_confirmed = False
                            reward = reward_from_events(events, episode_length, EPISODE_TIMEOUT * 60, track_direction, current_checkpoint_coverage, prev_checkpoint_coverage, car_speed, progress_delta, consecutive_checkpoints, car_center, mask_next, current_time)
                            
                            # Add to replay buffer with previous state and current reward
                            resized_prev_frame = cv2.resize(prev_frame, INPUT_SIZE)
                            prev_state = preprocess_frame(resized_prev_frame)
                            

                            # Check if episode should end
                            done = "finish" in events or "crash" in events or "stuck" in events or stuck_counter > 10
                            
                            store_experience_enhanced(replay_buffer, multi_step_buffer, prev_state, action_idx, reward, next_state, done, events)

                            if adaptive_trainer.should_train(step_idx):
                                loss = train_step_prioritized(q_net, target_net, optimizer, replay_buffer)
                                adaptive_trainer.update_frequency(loss)
                            
                            episode_reward += reward
                            reward_history.append(reward)
                        # Update target network more frequently
                        if step_idx % TARGET_UPDATE_FREQ == 0:
                            target_net.load_state_dict(q_net.state_dict())

                        # Slower epsilon decay
                        if epsilon > EPSILON_END:
                            epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY

                        # Store current state as previous for next iteration
                        prev_frame = frame_next.copy()
                        prev_action_idx = action_idx.copy()
                        prev_mask = mask_next.copy()
                        prev_checkpoint_coverage = current_checkpoint_coverage

                        
                        # Add action complexity penalty
                        # reward += action_penalty
                        
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
            stop_control_thread_func()  # Add this line
            release_all_keys()
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