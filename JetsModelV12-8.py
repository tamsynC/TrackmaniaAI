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
INPUT_SIZE = (256, 256)

ACTIONS = ['a', 'd', 'w', 's']
ACTION_INDEX = [0, 1, 2, 3]

NUM_ACTIONS = len(ACTIONS)

LABEL_BACKGROUND = 0
LABEL_CAR = 1
LABEL_TRACK = 4
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3


BATCH_SIZE = 64  
GAMMA = 0.99
LEARNING_RATE = 1e-4
REPLAY_BUFFER_CAPACITY = 5500    
TARGET_UPDATE_FREQ = 100  
EPSILON_START = 1.0
EPSILON_END = 0.03
EPSILON_DECAY = 500


EPISODE_TIMEOUT = 150
CRASH_THRESHOLD = 0.87
STUCK_THRESHOLD = 10
CHECKPOINT_CONFIRM_FRAMES = 2
FINISH_CONFIRM_FRAMES = 30  
VELOCITY_THRESHOLD = 3
FRAME_SKIP = 1
TRAIN_FREQUENCY = 1
CHECKPOINT_TIMEOUT = 100  
CHECKPOINT_TIMEOUT_PENALTY = -50  


CHECKPOINT_SAVE_FREQUENCY = 50  
MAX_CHECKPOINTS_TO_KEEP = 10
MIN_FREE_SPACE_GB = 1.0         


last_checkpoint_reach_time = None

action_queue = Queue()
control_thread = None
stop_control_thread = False
current_action = [0, 0, 0, 0]  
action_lock = threading.Lock()

crash_detection_active = False
crash_detection_thread = None
sustained_crash_detected = False
crash_penalty_accumulated = 0
crash_detection_lock = threading.Lock()



checkpoint_detection_queue = Queue()
checkpoint_detection_active = False
checkpoint_detection_thread = None
last_detected_checkpoints = []
checkpoint_event_queue = Queue()

   

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
    """Dedicated thread for handling WASD controls with short tap behavior"""
    global stop_control_thread, current_action

    prev_action_state = [False] * len(ACTIONS)

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

            # Only tap keys that are newly activated
            for i, should_press in enumerate(action_to_execute):
                key = ACTIONS[i]
                if key == "w":
                    if should_press and not prev_action_state[i]:
                        keyboard.press(key)
                        time.sleep(0.5)  # 10ms tap
                        keyboard.release(key)
                        prev_action_state[i] = True
                    elif not should_press and prev_action_state[i]:
                        # Mark as released, no need to press again
                        prev_action_state[i] = False


                elif should_press and not prev_action_state[i]:
                    keyboard.press(key)
                    time.sleep(0.1)  # 10ms tap
                    keyboard.release(key)
                    prev_action_state[i] = True
                elif not should_press and prev_action_state[i]:
                    # Mark as released, no need to press again
                    prev_action_state[i] = False

            # Small sleep to avoid tight loop hogging CPU
            time.sleep(0.001)

        except Exception as e:
            print(f"Control thread error: {e}")
            time.sleep(0.1)

    force_release_all_keys()



def force_release_all_keys():
    """Force release all keys using Windows API"""
    for key in ACTIONS:
        
        vk_code = ord(key.upper())
        
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.005)



def train_dqn(q_net, target_net, optimizer, replay_buffer):
    """Simple DQN training step"""
    batch = replay_buffer.sample(BATCH_SIZE)
    if batch is None:
        return None
    
    states, actions, rewards, next_states, dones = batch
    
    states = torch.FloatTensor(states).to(device)
    actions = torch.FloatTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.BoolTensor(dones).to(device)
    
    # Current Q values
    current_q_values = q_net(states)
    current_q_values = (current_q_values * actions).sum(1)
    
    # Next Q values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + (GAMMA * next_q_values * ~dones)
    
    # Loss
    loss = F.mse_loss(current_q_values, target_q_values)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CODEITERATION11")
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
                
                episode_num = int(file.replace(pattern, "").replace(".pth", ""))
                checkpoint_files.append((episode_num, file))
            except ValueError:
                continue
    
    
    checkpoint_files.sort(reverse=True)
    
    
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


frame_cache = {}
cache_lock = threading.Lock()

# 1. OPTIMIZE FRAME PROCESSING - BIGGEST BOTTLENECK

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
            if len(frame_cache) > 10:  
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
    
    
    diff = np.mean((f1.astype(np.float32) - f2.astype(np.float32)) ** 2)
    similarity = 1.0 / (1.0 + diff / 1000.0)  
    return similarity >= threshold

def calculate_track_direction(mask):
    """Optimized track direction calculation"""
    car_mask = (mask == LABEL_CAR)
    track_mask = (mask == LABEL_TRACK)
    
    car_center = get_centroid(car_mask)
    if car_center is None:
        return 0
    
    
    track_ys, track_xs = np.where(track_mask)
    if len(track_xs) == 0:
        return 0
    
    
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
    
    
    current_pos = np.array(current_pos, dtype=np.float32)
    prev_pos = np.array(prev_pos, dtype=np.float32)
    
    
    distance = np.linalg.norm(current_pos - prev_pos)
    
    
    speed = distance / dt
    
    
    speed = np.clip(speed, 0, 500)  
    
    return float(speed)


def calculate_progress(car_center, checkpoint_mask, finish_mask):
    """Calculate rough progress through the track"""
    if car_center is None:
        return 0
    
    
    checkpoint_pixels = np.sum(checkpoint_mask)
    finish_pixels = np.sum(finish_mask)
    
    
    
    progress = (checkpoint_pixels + finish_pixels * 2) / 100  
    return progress


def enhanced_checkpoint_buffer_detection(mask, car_center, checkpoint_history_buffer, detection_threshold=0.03):
    """More reliable checkpoint detection"""
    checkpoint_mask = (mask == LABEL_CHECKPOINT)
    current_checkpoint_pixels = np.sum(checkpoint_mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    checkpoint_coverage = current_checkpoint_pixels / total_pixels
    
    
    checkpoint_data = {
        'coverage': checkpoint_coverage,
        'car_center': car_center,
        'timestamp': time.time(),
        'checkpoint_pixels': current_checkpoint_pixels
    }
    
    checkpoint_history_buffer.append(checkpoint_data)
    
    
    if len(checkpoint_history_buffer) > 10:
        checkpoint_history_buffer.popleft()
    
    
    checkpoint_detected = checkpoint_coverage > detection_threshold
    
    
    
    
    return checkpoint_detected, checkpoint_coverage

def start_checkpoint_detection_thread():
    """Start dedicated checkpoint detection thread"""
    global checkpoint_detection_thread, checkpoint_detection_active
    
    checkpoint_detection_active = True
    checkpoint_detection_thread = threading.Thread(target=checkpoint_detection_worker, daemon=True)
    checkpoint_detection_thread.start()
    print("🎯 Checkpoint detection thread started")

def stop_checkpoint_detection_thread():
    """Stop checkpoint detection thread"""
    global checkpoint_detection_thread, checkpoint_detection_active
    
    checkpoint_detection_active = False
    if checkpoint_detection_thread and checkpoint_detection_thread.is_alive():
        checkpoint_detection_thread.join(timeout=1.0)
    print("🛑 Checkpoint detection thread stopped")

def checkpoint_detection_worker():
    """Worker thread for continuous checkpoint monitoring"""
    global checkpoint_detection_active, last_detected_checkpoints
    
    checkpoint_buffer = deque(maxlen=20)  
    
    while checkpoint_detection_active:
        try:
            
            if not checkpoint_detection_queue.empty():
                checkpoint_data = checkpoint_detection_queue.get_nowait()
                checkpoint_buffer.append(checkpoint_data)
                
                
                if len(checkpoint_buffer) >= 5:
                    
                    coverages = [data['coverage'] for data in checkpoint_buffer]
                    
                    
                    if len(coverages) >= 5:
                        recent_peak = max(coverages[-5:])
                        if recent_peak > 0.03:  
                            
                            current_coverage = coverages[-1]
                            if recent_peak - current_coverage > 0.01:  
                                print(f"🎯 Thread detected checkpoint passage! Peak: {recent_peak:.3f}")
                                checkpoint_event_queue.put("checkpoint_reached")
                                
                                
                
        except Exception as e:
            print(f"Checkpoint detection thread error: {e}")
        
        time.sleep(0.01)  



def check_checkpoint_timeout(last_checkpoint_time, current_time):
    """Check if too much time has passed since last checkpoint"""
    if last_checkpoint_time is None:
        return False
    return (current_time - last_checkpoint_time) > CHECKPOINT_TIMEOUT

def simple_checkpoint_detection(mask, prev_checkpoint_pixels=0):
    """Simple checkpoint detection based on screen coverage"""
    checkpoint_mask = (mask == LABEL_CHECKPOINT)
    current_checkpoint_pixels = np.sum(checkpoint_mask)
    
    
    total_pixels = mask.shape[0] * mask.shape[1]
    checkpoint_ratio = current_checkpoint_pixels / total_pixels
    
    threshold = 0.07  
    
    if checkpoint_ratio > threshold:
        print(f"Checkpoint detected! Screen coverage: {checkpoint_ratio:.3f} ({checkpoint_ratio*100:.1f}%)")
        return True
    
    return False

def detect_track_coverage_crash(mask, threshold=0.2):
    """Start crash detection in a separate thread"""
    global crash_detection_active, crash_detection_thread, sustained_crash_detected, crash_penalty_accumulated
    
    
    track_mask = (mask == LABEL_TRACK)
    track_pixels = np.sum(track_mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    coverage = track_pixels / total_pixels
    
    
    if coverage >= threshold:
        with crash_detection_lock:
            crash_detection_active = False
        return False, 0
    
    
    with crash_detection_lock:
        if not crash_detection_active and not sustained_crash_detected:
            crash_detection_active = True
            crash_penalty_accumulated = 0
            crash_detection_thread = threading.Thread(
                target=crash_detection_worker, 
                args=(threshold,), 
                daemon=True
            )
            crash_detection_thread.start()
    
    
    with crash_detection_lock:
        
        return sustained_crash_detected, crash_penalty_accumulated

def crash_detection_worker(threshold):
    """Worker thread for crash detection timing"""
    global crash_detection_active, sustained_crash_detected, crash_penalty_accumulated
    
    start_time = time.time()
    
    while True:
        with crash_detection_lock:
            if not crash_detection_active:
                break
        elapsed_time = time.time() - start_time
        
        
        if elapsed_time >= 0.3 and crash_penalty_accumulated == 0:
            with crash_detection_lock:
                crash_penalty_accumulated = -8
        
        
        if elapsed_time >= 2.5:
            with crash_detection_lock:
                sustained_crash_detected = True
                
                print(f"🚨 Sustained track coverage crash detected (under {threshold*100:.1f}%) for 1.5s!")
                crash_detection_active = False
                
                break
        
        time.sleep(0.05)

def enhanced_checkpoint_detection(mask, prev_checkpoint_coverage=0, checkpoint_counter=0, checkpoint_confirmed=False):
    """Enhanced checkpoint detection with progressive rewards"""
    checkpoint_mask = (mask == LABEL_CHECKPOINT)
    car_mask = (mask == LABEL_CAR)
    
    current_checkpoint_pixels = np.sum(checkpoint_mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    checkpoint_coverage = current_checkpoint_pixels / total_pixels
    
    events = []
    

    if checkpoint_coverage > 0.001:  
        events.append("checkpoint_visible")
    
    if checkpoint_coverage > 0.03:   
        events.append("checkpoint_approaching")
    
    if checkpoint_coverage > 0.07:   
        checkpoint_counter += 1
        if checkpoint_counter >= CHECKPOINT_CONFIRM_FRAMES and not checkpoint_confirmed:
            events.append("checkpoint_reached")
            checkpoint_confirmed = True
            print(f"Checkpoint reached! Coverage: {checkpoint_coverage:.3f} ({checkpoint_coverage*100:.1f}%)")
    else:
        checkpoint_counter = 0
        checkpoint_confirmed = False

    
    return events, checkpoint_coverage, checkpoint_counter, checkpoint_confirmed





similarity_history = deque(maxlen=7)  

def detect_events(mask, prev_mask, f2, prev_positions, stuck_counter, checkpoint_counter, 
                 checkpoint_confirmed, finish_counter, finish_confirmed, last_checkpoint_time, 
                 current_time, prev_checkpoint_coverage=0, checkpoint_history_buffer=None):
    """Fixed event detection function"""
    global sustained_crash_detected
    
    car_mask = (mask == LABEL_CAR)
    track_mask = (mask == LABEL_TRACK)
    checkpoint_mask = (mask == LABEL_CHECKPOINT)
    finish_mask = (mask == LABEL_FINISH)

    car_bbox = get_bbox(car_mask)
    car_center = get_centroid(car_mask)
    track_bbox = get_bbox(track_mask)

    events = []

    
    if checkpoint_history_buffer is None:
        checkpoint_history_buffer = deque(maxlen=10)
    
    
    checkpoint_detected, current_checkpoint_coverage = enhanced_checkpoint_buffer_detection(
        mask, car_center, checkpoint_history_buffer
    )
    
    
    if checkpoint_detection_active:
        checkpoint_data = {
            'coverage': current_checkpoint_coverage,
            'car_center': car_center,
            'timestamp': current_time,
            'checkpoint_pixels': np.sum(checkpoint_mask)
        }
        try:
            checkpoint_detection_queue.put_nowait(checkpoint_data)
        except:
            pass
    
    
    try:
        while not checkpoint_event_queue.empty():
            thread_event = checkpoint_event_queue.get_nowait()
            if thread_event == "checkpoint_reached":
                checkpoint_detected = True
                print("🎯 Thread detected checkpoint!")
    except:
        pass
    
    
    if checkpoint_detected:
        checkpoint_counter += 1
        if checkpoint_counter >= CHECKPOINT_CONFIRM_FRAMES and not checkpoint_confirmed:
            events.append("checkpoint")  
            checkpoint_confirmed = True
            last_checkpoint_reach_time = current_time  
            print(f"✅ Checkpoint confirmed! Coverage: {current_checkpoint_coverage:.3f}")
    else:
        if checkpoint_counter > 0:
            checkpoint_counter = max(0, checkpoint_counter - 1)  
        if checkpoint_counter == 0:
            checkpoint_confirmed = False

    
    if car_bbox is None or (track_bbox is not None and not boxes_overlap(car_bbox, track_bbox)):
        events.append("out_of_bounds")

    
    if np.any(finish_mask & car_mask):
        finish_counter += 1
        if finish_counter >= FINISH_CONFIRM_FRAMES and not finish_confirmed:
            events.append("finish")
            finish_confirmed = True
            print("🏁 FINISH LINE REACHED!")
    else:
        finish_counter = max(0, finish_counter - 1)  
        if finish_counter == 0:
            finish_confirmed = False

    
    if check_checkpoint_timeout(last_checkpoint_time, current_time):
        events.append("checkpoint_timeout")

    
    if current_time - last_checkpoint_time > 5 and prev_mask is not None:
        similarity = np.mean(prev_mask == mask)

        similarity_history.append(similarity > 0.98)

        if similarity_history.count(True) >= 5: 
            events.append("similarity_crash")
            # events.append("crash")
            print("💥 Similarity crash detected")
            similarity_history.clear()  


        
        
        
        
        
        
        
        
        

    
    sustained_crash, crash_penalty = detect_track_coverage_crash(mask)

    if sustained_crash:
        events.append("crash")

    
    
    
    
    
    
    
    
    
    

    
    return (events, stuck_counter + 1 if "stuck" in events else 0, 
            checkpoint_counter, checkpoint_confirmed, finish_counter, finish_confirmed, 
            last_checkpoint_time, current_checkpoint_coverage, crash_penalty)

def reward_from_events(events, episode_length, max_episode_length, track_direction, 
                      checkpoint_coverage=0, prev_checkpoint_coverage=0, car_speed=0, 
                      consecutive_checkpoints=0, car_center=None, 
                      mask=None, current_time=None, crash_penalty=0, last_checkpoint_time=None):
    """Fixed reward function that always returns a value"""
    global checkpoint_start_time, last_checkpoint_reach_time
    
    reward = 0
    checkpointhresholdtime = 0.5
    
    if last_checkpoint_time is not None and current_time is not None:
        time_since_last_checkpoint = current_time - last_checkpoint_time
    # print(time_since_last_checkpoint)
    if time_since_last_checkpoint > 12:
        reward -= 10
    if "finish" in events:
        reward += 1000
        print(f"🏆 RACE COMPLETED! +1000")
        
    elif "checkpoint" in events:
        base_checkpoint_reward = 25
        consecutive_bonus = consecutive_checkpoints * 25
        
        
        time_bonus = 0

            # print(time_since_last_checkpoint)
        if time_since_last_checkpoint < checkpointhresholdtime:
            time_bonus = -5
            print("⚠️ Checkpoint too fast (possible cheat)")
        elif time_since_last_checkpoint < 1.5:
            time_bonus = 70
            print(f"🚀 LIGHTNING FAST! +{time_bonus}")
        elif time_since_last_checkpoint < 3.0:
            time_bonus = 40
            print(f"⚡ VERY FAST! +{time_bonus}")
        elif time_since_last_checkpoint < 5.0:
            time_bonus = 30
            print(f"🏃 FAST! +{time_bonus}")
        elif time_since_last_checkpoint < 6.0:
            time_bonus = 20
            print(f"👍 GOOD TIME! +{time_bonus}")
        else:
            time_bonus = 0
            print(f"🐌 Slow checkpoint...")
        
        total_checkpoint_reward = base_checkpoint_reward + consecutive_bonus + time_bonus
        reward += total_checkpoint_reward
        print(f"✅ Checkpoint reward: {total_checkpoint_reward} (base:{base_checkpoint_reward} + consecutive:{consecutive_bonus} + speed:{time_bonus})")

    
    if "crash" in events:
        crash_penalty_val = 10 - consecutive_checkpoints * 2
        reward -= crash_penalty_val
        print(f"💥 Crash penalty: -{crash_penalty_val}")
        
    if "similarity_crash" in events:
        reward -= 10
        print(f"💥 Similarity crash penalty: -30")
        
    if "stuck" in events:
        stuck_penalty = -15 - consecutive_checkpoints * 3
        reward -= stuck_penalty
        print(f"🚫 Stuck penalty: -{stuck_penalty}")
    
    if "out_of_bounds" in events:
        oob_penalty = 7 - consecutive_checkpoints
        reward -= oob_penalty
        print(f"🚫 Out of bounds penalty: -{oob_penalty}")
    
    if "checkpoint_timeout" in events:
        reward -= 5
        print(f"⏰ Checkpoint timeout penalty: -5")

    
    if track_direction > -0.7 and track_direction < 0.7:
        direction_reward = 5 - (abs(track_direction) * 4)
        reward += direction_reward
    
    
    reward += crash_penalty
    
    
    consecutive_bonus = 1 * consecutive_checkpoints
    reward += consecutive_bonus
    
    print(f"📊 Total reward: {reward:.2f}")
    return reward

class QNetwork(nn.Module):
    def __init__(self, input_channels=3, num_actions=4):
        super(QNetwork, self).__init__()
        
        # Simple CNN
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened features
        conv_out_size = self._get_conv_out_size(input_channels, INPUT_SIZE[0], INPUT_SIZE[1])
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def _get_conv_out_size(self, channels, height, width):
        with torch.no_grad():
            x = torch.zeros(1, channels, height, width)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.size()))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def press_action(action_idx):
    """Queue action for the dedicated control thread"""
    global action_queue

    try:
        # Clear old actions
        with action_queue.mutex:
            action_queue.queue.clear()

        # Add new one
        action_queue.put_nowait(action_idx.copy())
    except Exception as e:
        print(f"Error queuing action: {e}")

def release_all_keys():
    for key in ACTIONS:
        keyboard.release(key)

def restart_track():
    print("🔄 Restarting track...")
    global crash_detection_active, sustained_crash_detected
    with crash_detection_lock:
        crash_detection_active = False
        sustained_crash_detected = False
    
    while not action_queue.empty():
        try:
            action_queue.get_nowait()
            action_queue.task_done()
        except:
            break
    
    
    with action_lock:
        current_action = [0, 0, 0, 0]
    
    force_release_all_keys()
    time.sleep(0.005)
    
    for _ in range(3):
        try:
            keyboard.press_and_release('Backspace')
            break
        except:
            time.sleep(0.01)
    
    
    force_release_all_keys()
    time.sleep(2)

def select_action(q_net, state, epsilon, track_direction):
    """Select action using epsilon-greedy with track direction bias"""
    if random.random() < epsilon:
        # Random action with some bias towards forward movement
        action = [0, 0, 0, 0]  # [left, right, forward, backward]
        
        # Always tend to go forward
        if random.random() < 0.3:
            action[2] = 1  # forward
        
        # Turn based on track direction
        if track_direction > 0.3:  # track curves right
            if random.random() < 0.1:
                action[1] = 1  # right
        elif track_direction < -0.3:  # track curves left
            if random.random() < 0.1:
                action[0] = 1  # left
        else:
            # Slight random turning
            if random.random() < 0.1:
                action[random.choice([0, 1])] = 1
        
        return action
    else:
        # Use Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_net(state_tensor)
            # Convert Q-values to probabilities using sigmoid
            action_probs = torch.sigmoid(q_values.squeeze())

            # Extract individual action probabilities
            left_prob = action_probs[0].item()
            right_prob = action_probs[1].item()

            # Default thresholded actions
            action = [0] * len(action_probs)

            # Handle turning logic (assuming index 0 = left, 1 = right)
            if left_prob > 0.99 and right_prob > 0.99:
                if left_prob > right_prob:
                    action[0] = 1  # turn left
                else:
                    action[1] = 1  # turn right
            else:
                if left_prob > 0.99:
                    action[0] = 1
                if right_prob > 0.99:
                    action[1] = 1

            # Handle the rest of the actions normally (e.g., throttle, brake, etc.)
            for i in range(2, len(action_probs)):
                if action_probs[i] > 0.98:
                    action[i] = 1

            print(action_probs)
            print(action)
        return action

    
def preprocess_frame(frame):

    if len(frame.shape) == 3 and frame.shape[2] == 3:  
        tensor = transform(frame).float()  
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    
    return tensor.numpy()

def initialize_checkpoint_detection():
    
    checkpoint_history_buffer = deque(maxlen=10)
    start_checkpoint_detection_thread()
    return checkpoint_history_buffer

class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.bool_))
    
    def __len__(self):
        return len(self.buffer)
def main():
    free_space = check_disk_space()
    print(f"💾 Available disk space: {free_space:.2f}GB")
    reward_history = []

    if free_space < MIN_FREE_SPACE_GB * 2:  
        print("⚠️ Warning: Low disk space! Consider freeing up space before training.")
        cleanup_old_checkpoints()
    
    print("🔧 Loading segmentation model...")



    seg_model = load_segmentation_model(MODEL_PATH)
    start_control_thread()
    checkpoint_history_buffer = initialize_checkpoint_detection()
    

    # Initialize networks
    q_net = QNetwork(input_channels=3, num_actions=NUM_ACTIONS).to(device)
    target_net = QNetwork(input_channels=3, num_actions=NUM_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    
    # Enable optimizations
    # if device.type == 'cuda':
    #     q_net = q_net.half()
    #     target_net = target_net.half()
    
    # Compile networks
    # try:
    #     q_net = torch.compile(q_net, mode='reduce-overhead')
    #     target_net = torch.compile(target_net, mode='reduce-overhead')
    #     print("✅ Networks compiled for optimization")
    # except:
    #     print("⚠️ Network compilation not available")
    
    optimizer = optim.AdamW(q_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    replay_buffer = SimpleReplayBuffer(REPLAY_BUFFER_CAPACITY)

    epsilon = EPSILON_START
    step_idx = 0

    monitor = find_trackmania_window()
    print(f"🎮 Capturing Trackmania window at: {monitor}")

    total_episodes = 300
    prev_mask = None
    
    with mss.mss() as sct:
        try:
            for episode in range(total_episodes):
                episode_reward = 0
                episode_length = 0
                episode_start_time = time.time()
                
                # Initialize episode variables
                prev_positions = deque(maxlen=STUCK_THRESHOLD)
                stuck_counter = 0
                checkpoint_counter = 0
                checkpoint_confirmed = False
                finish_counter = 0
                finish_confirmed = False
                frame_count = 0
                consecutive_checkpoints = 0
                max_consecutive_checkpoints = 0
                prev_checkpoint_coverage = 0
                current_checkpoint_coverage = 0
                prev_frame = None
                prev_car_pos = None
                last_checkpoint_time = time.time()
                last_checkpoint_reach_time = time.time()
                
                # FIXED: Initialize state variables for replay buffer
                prev_state = None
                prev_action = None
                
                print(f"\n🚗 Starting Episode {episode + 1}/{total_episodes} (ε={epsilon:.3f})")
                
                while True:
                    current_time = time.time()
                    
                    # Episode timeout check
                    if current_time - episode_start_time > EPISODE_TIMEOUT:
                        print("⏰ Episode timeout - ending episode")
                        episode_reward -= 100
                        break
                    
                    # Periodic key release to prevent stuck keys
                    if episode_length % 100 == 0:
                        force_release_all_keys()
                        time.sleep(0.005)
                    
                    frame_count += 1
                    
                    try:
                        # Capture frame
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
                            print(f"Segmentation error: {e}")
                            continue

                        # Calculate game state information
                        track_direction = calculate_track_direction(mask)
                        car_mask = (mask == LABEL_CAR)
                        car_center = get_centroid(car_mask)
                        car_speed = calculate_car_speed(car_center, prev_car_pos) if prev_car_pos else 0
                        
                        # Select and execute action
                        action_idx = select_action(q_net, state, epsilon, track_direction)
                        press_action(action_idx)
                        time.sleep(0.005)
                        
                        # Capture next frame
                        sct_img_next = sct.grab(monitor)
                        frame_next = np.array(sct_img_next)[..., :3]
                        frame_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2RGB)

                        try:
                            mask_next, resized_frame_next = segment_frame(seg_model, frame_next)
                            next_state = preprocess_frame(resized_frame_next)
                        except Exception as e:
                            print(f"Next frame segmentation error: {e}")
                            continue
                        
                        reward = 0
                        events = []
                        
                        # Event detection and reward calculation
                        if prev_frame is not None:
                            events, stuck_counter, checkpoint_counter, checkpoint_confirmed, finish_counter, finish_confirmed, last_checkpoint_time, current_checkpoint_coverage, crash_penalty = detect_events(
                                mask_next, prev_mask, frame_next, prev_positions, stuck_counter, 
                                checkpoint_counter, checkpoint_confirmed, finish_counter, finish_confirmed, 
                                last_checkpoint_time, current_time, prev_checkpoint_coverage, checkpoint_history_buffer)

                            reward = reward_from_events(
                                events, episode_length, EPISODE_TIMEOUT * 60, track_direction, 
                                current_checkpoint_coverage, prev_checkpoint_coverage, car_speed, 
                                consecutive_checkpoints, car_center, mask_next, 
                                current_time, crash_penalty, last_checkpoint_time
                            )
                            
                            # FIXED: Store experience in replay buffer with correct variables
                            if prev_state is not None and prev_action is not None:
                                # Determine if this is a terminal state
                                done = "finish" in events
                                
                                # Add experience to replay buffer
                                replay_buffer.push(prev_state, prev_action, reward, state, done)
                                
                                # Train the network
                                if len(replay_buffer) > BATCH_SIZE and step_idx % TRAIN_FREQUENCY == 0:
                                    loss = train_dqn(q_net, target_net, optimizer, replay_buffer)
                                    if loss is not None and step_idx % 100 == 0:
                                        print(f"📈 Training loss: {loss:.4f}")
                        
                        # Update episode statistics
                        episode_reward += reward
                        reward_history.append(reward)
                        
                        # Handle different events
                        if "checkpoint" in events:
                            consecutive_checkpoints += 1
                            max_consecutive_checkpoints = max(max_consecutive_checkpoints, consecutive_checkpoints)
                            last_checkpoint_time = time.time()
                            print(f"✅ Checkpoint! Total checkpoints: {consecutive_checkpoints}, Reward: {reward:.1f}")
                        
                        # FIXED: Proper crash handling - restart but continue episode
                        if "crash" in events or "stuck" in events or stuck_counter > STUCK_THRESHOLD:
                            print(f"💥 {'Crash' if 'crash' in events else 'Stuck'} detected - restarting track")
                            restart_track()
                            consecutive_checkpoints = 0
                            max_consecutive_checkpoints = 0
                            # Reset some counters but DON'T break the episode
                            stuck_counter = 0
                            checkpoint_counter = 0
                            checkpoint_confirmed = False
                            finish_counter = 0
                            finish_confirmed = False
                            last_checkpoint_time = time.time()
                            # Clear frame cache and wait a moment
                            prev_frame = None
                            prev_mask = None
                            target_net.load_state_dict(q_net.state_dict())
                            print(f"🔄 Target network updated at step {step_idx} Epsilon: {epsilon}")
                            continue  # Continue the episode, don't break
                        
                        # Handle race finish
                        if "finish" in events:
                            print(f"🏁 Race completed! Episode reward: {episode_reward:.2f}")
                            break  # End episode on finish
                        
                        # Update target network periodically
                        # if step_idx % TARGET_UPDATE_FREQ == 0:
                        #     target_net.load_state_dict(q_net.state_dict())
                        #     print(f"🔄 Target network updated at step {step_idx}")
                        
                        # Decay epsilon
                        if epsilon > EPSILON_END:
                            epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY
                        
                        # Update state variables for next iteration
                        prev_state = state
                        prev_action = action_idx
                        prev_frame = frame_next.copy()
                        prev_car_pos = car_center
                        prev_mask = mask_next.copy()
                        prev_checkpoint_coverage = current_checkpoint_coverage
                        
                        step_idx += 1
                        episode_length += 1

                    except Exception as e:
                        print(f"Frame processing error: {e}")
                        continue

                print(f"📊 Episode {episode + 1} - Reward: {episode_reward:.2f}, Steps: {episode_length}, ε: {epsilon:.3f}, Max Checkpoints: {max_consecutive_checkpoints}")
                
                # Save checkpoint periodically
                if (episode + 1) % CHECKPOINT_SAVE_FREQUENCY == 0:
                    checkpoint_filename = f"trackmania_dqn_checkpoint_{episode + 1}.pth"
                    if safe_save_model(q_net.state_dict(), checkpoint_filename):
                        cleanup_old_checkpoints()
                    
                    # Clear frame cache
                    with cache_lock:
                        frame_cache.clear()

            print("🎉 Training completed!")
            
            # Save final model
            if not safe_save_model(q_net.state_dict(), "trackmania_dqn_final_optimized.pth"):
                print("⚠️ Could not save final model due to disk space issues")

        except KeyboardInterrupt:
            print("🛑 Training interrupted by user.")
        finally:
            # Cleanup
            with crash_detection_lock:
                crash_detection_active = False
            stop_checkpoint_detection_thread()
            stop_control_thread_func()
            release_all_keys()
            
            # Plot reward history
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