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

NUM_ACTIONS = len(ACTIONS)

LABEL_BACKGROUND = 0
LABEL_CAR = 1
LABEL_TRACK = 4
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3


BATCH_SIZE = 64  
GAMMA = 0.95
LEARNING_RATE = 1e-4
REPLAY_BUFFER_CAPACITY = 200000    
TARGET_UPDATE_FREQ = 100  
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000 


EPISODE_TIMEOUT = 300
CRASH_THRESHOLD = 0.87
STUCK_THRESHOLD = 10
CHECKPOINT_CONFIRM_FRAMES = 2
FINISH_CONFIRM_FRAMES = 30  
VELOCITY_THRESHOLD = 3
FRAME_SKIP = 4  
TRAIN_FREQUENCY = 1  
CHECKPOINT_TIMEOUT = 100  
CHECKPOINT_TIMEOUT_PENALTY = -500  



CHECKPOINT_SAVE_FREQUENCY = 50  
MAX_CHECKPOINTS_TO_KEEP = 3     
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


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  
        self.beta = beta    
        self.beta_increment = beta_increment
        

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
            

        priorities = np.array(self.priorities)
        

        probs = priorities ** self.alpha
        probs /= probs.sum()
        

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones, events_list = zip(*experiences)
        
        
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  
        
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.uint8), 
                weights, indices)
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha  
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
            
            try:
                new_action = action_queue.get_nowait()
                with action_lock:
                    current_action = new_action
                action_queue.task_done()
            except:
                pass  
            
            
            with action_lock:
                action_to_execute = current_action.copy()
            
            
            for i, should_press in enumerate(action_to_execute):
                
                if should_press:
                    keyboard.press(ACTIONS[i])
                else:
                    keyboard.release(ACTIONS[i])
            
            time.sleep(0.1)  
            
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



def train_step_prioritized(q_net, target_net, optimizer, replay_buffer):
    """Enhanced training with prioritized experience replay"""
    if len(replay_buffer) < BATCH_SIZE:
        return None
    
    
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
    
    
    q_values = q_net(states_v)
    state_action_values = (q_values * actions_v).sum(1)
    
    
    with torch.no_grad():
        
        next_q_main = q_net(next_states_v)
        next_actions = next_q_main.max(1)[1].unsqueeze(1)
        
        
        next_q_target = target_net(next_states_v)
        next_q_max = next_q_target.gather(1, next_actions).squeeze()
        
        expected_q_values = rewards_v + (GAMMA * next_q_max * (1 - dones_v))
    
    
    td_errors = (expected_q_values - state_action_values).detach().cpu().numpy()
    
    
    loss = (weights_v * F.smooth_l1_loss(state_action_values, expected_q_values, reduction='none')).mean()
    
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    optimizer.step()
    
    
    replay_buffer.update_priorities(indices, td_errors)
    
    return loss.item()


def calculate_experience_bonus(events, replay_buffer):
    """Give bonus for rare/important experiences"""
    bonus = 0
    
    
    if len(replay_buffer) > 1000:
        recent_experiences = list(replay_buffer.buffer)[-1000:]  
        
        checkpoint_count = sum(1 for exp in recent_experiences if exp[5] and "checkpoint" in exp[5])
        finish_count = sum(1 for exp in recent_experiences if exp[5] and "finish" in exp[5])
        
        
        if "checkpoint" in events:
            rarity_bonus = max(0, 100 - checkpoint_count * 10)  
            bonus += rarity_bonus
            
        if "finish" in events:
            rarity_bonus = max(0, 500 - finish_count * 50)  
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
            
            n_step_return = 0
            for i, (_, _, r, _, d) in enumerate(self.buffer):
                n_step_return += (self.gamma ** i) * r
                if d:  
                    break
            
            
            first_state, first_action = self.buffer[0][:2]
            last_next_state, last_done = self.buffer[-1][3:]
            
            return (first_state, first_action, n_step_return, last_next_state, last_done)
        
        return None
    
    def clear(self):
        self.buffer.clear()








def initialize_enhanced_replay_system():
    """Initialize the enhanced replay system"""
    
    replay_buffer = PrioritizedReplayBuffer(
        capacity=REPLAY_BUFFER_CAPACITY,
        alpha=0.6,      
        beta=0.4,       
        beta_increment=0.001  
    )
    
    
    multi_step_buffer = MultiStepBuffer(n_steps=3, gamma=GAMMA)
    
    return replay_buffer, multi_step_buffer


def store_experience_enhanced(replay_buffer, multi_step_buffer, prev_state, action_idx, reward, next_state, done, events):
    """Enhanced experience storage with multi-step learning"""
    
    
    experience_bonus = calculate_experience_bonus(events, replay_buffer)
    enhanced_reward = reward + experience_bonus
    
    
    n_step_transition = multi_step_buffer.add(prev_state, action_idx, enhanced_reward, next_state, done)
    
    
    if n_step_transition is not None:
        state, action, n_step_return, final_next_state, final_done = n_step_transition
        replay_buffer.push(state, action, n_step_return, final_next_state, final_done, events)
    
    
    replay_buffer.push(prev_state, action_idx, enhanced_reward, next_state, done, events)
    
    
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
                
                
                if recent_loss < older_loss * 0.9:
                    self.current_freq = min(self.max_freq, self.current_freq + 1)
                
                elif recent_loss > older_loss * 1.1:
                    self.current_freq = max(self.min_freq, self.current_freq - 1)








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
        
        
        if elapsed_time >= 1.5:
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

        similarity_history.append(similarity > 0.987)

        if similarity_history.count(True) >= 5: 
            events.append("similarity_crash")
            events.append("crash")
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
    checkpointhresholdtime = 1
    
    
    
    
    if "finish" in events:
        reward += 1000
        print(f"🏆 RACE COMPLETED! +1000")
        
    elif "checkpoint" in events:
        base_checkpoint_reward = 25
        consecutive_bonus = consecutive_checkpoints * 25
        
        
        time_bonus = 0
        if last_checkpoint_time is not None and current_time is not None:
            time_since_last_checkpoint = current_time - last_checkpoint_time
            print(time_since_last_checkpoint)
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
        crash_penalty_val = 70 - consecutive_checkpoints * 15
        reward -= crash_penalty_val
        print(f"💥 Crash penalty: -{crash_penalty_val}")
        
    if "similarity_crash" in events:
        reward -= 50
        print(f"💥 Similarity crash penalty: -30")
        
    if "stuck" in events:
        stuck_penalty = 30 - consecutive_checkpoints * 10
        reward -= stuck_penalty
        print(f"🚫 Stuck penalty: -{stuck_penalty}")
    
    if "out_of_bounds" in events:
        oob_penalty = 40 - consecutive_checkpoints * 10
        reward -= oob_penalty
        print(f"🚫 Out of bounds penalty: -{oob_penalty}")
    
    if "checkpoint_timeout" in events:
        reward -= 5
        print(f"⏰ Checkpoint timeout penalty: -5")

    
    if track_direction > -0.7 and track_direction < 0.7:
        direction_reward = 5 - (abs(track_direction) * 5)
        reward += direction_reward
    
    
    reward += crash_penalty
    
    
    consecutive_bonus = 1 * consecutive_checkpoints
    reward += consecutive_bonus
    
    print(f"📊 Total reward: {reward:.2f}")
    return reward


class QNetwork(nn.Module):
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

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
def press_action(action_idx):
    """Queue action for the dedicated control thread"""
    global action_queue
    
    try:
        
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

def select_action(q_net, state, epsilon, track_direction, device='cuda'):
    if random.random() < epsilon:
        action_idx = [0, 0, 0, 0]
        r = 0.2 

        if random.random() < 0.02:
            return [random.randint(0, 1) for _ in range(4)]


        turn_prob = min(max(abs(track_direction), 0.1), 0.4)  

        if track_direction > 0.2:  
            if random.random() < turn_prob:
                action_idx[1] = 1  
            elif random.random() < r:
                action_idx[0] = 1
        elif track_direction < -0.2:  
            if random.random() < turn_prob:
                action_idx[0] = 1  
            elif random.random() < r:
                action_idx[1] = 1
        else:
            turn_decision = random.random()
            if turn_decision < r/2:
                action_idx[random.choice([0, 1])] = 1



        
        if random.random() < 0.9:
            action_idx[2] = 1  
        elif random.random() < 0.1:
            action_idx[3] = 1  

        
        if random.random() < 0.2:
            return [0, 0, 0, 0]
        

        return action_idx

    else:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_net(state_t)
            action_probs = torch.sigmoid(q_values.squeeze())

            action_idx = [
                int(action_probs[0].item() > 0.5),  
                int(action_probs[1].item() > 0.5),  
                int(action_probs[2].item() > 0.4),  
                int(action_probs[3].item() > 0.6),  
            ]

        return action_idx


    
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
    q_net = QNetwork((3, INPUT_SIZE[0], INPUT_SIZE[1]), NUM_ACTIONS).to(device)  
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

    total_episodes = 300
    prev_mask = None
    
    with mss.mss() as sct:
        try:
            for episode in range(total_episodes):
                events = []
                episode_reward = 0
                episode_length = 0
                episode_start_time = time.time()
                reward = 0
                crash_penalty = 0
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
                last_checkpoint_time = time.time()  
                last_checkpoint_reach_time = time.time()
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
                        
                        
                        car_mask = (mask == LABEL_CAR)
                        car_center = get_centroid(car_mask)
                        car_speed = calculate_car_speed(car_center, prev_car_pos) if prev_car_pos else 0
                        
                        checkpoint_mask = (mask == LABEL_CHECKPOINT)
                        finish_mask = (mask == LABEL_FINISH)
                        current_progress = calculate_progress(car_center, checkpoint_mask, finish_mask)
                        
                        
                        action_idx = select_action(q_net, state, epsilon, track_direction)

                        press_action(action_idx)

                        time.sleep(0.005)  
                        sct_img_next = sct.grab(monitor)
                        frame_next = np.array(sct_img_next)[..., :3]
                        frame_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2RGB)

                        try:
                            mask_next, resized_frame_next = segment_frame(seg_model, frame_next)
                            next_state = preprocess_frame(resized_frame_next)
                        except Exception as e:
                            continue
                        
                        done = False


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
                            
      
                            
                            
                            resized_prev_frame = cv2.resize(prev_frame, INPUT_SIZE)
                            prev_state = preprocess_frame(resized_prev_frame)
                            

                            
                            done = "finish" in events or "crash" in events
                            
                            store_experience_enhanced(replay_buffer, multi_step_buffer, prev_state, action_idx, reward, next_state, done, events)

                            if adaptive_trainer.should_train(step_idx):
                                loss = train_step_prioritized(q_net, target_net, optimizer, replay_buffer)
                                adaptive_trainer.update_frequency(loss)
                            
                            episode_reward += reward
                            reward_history.append(reward)







                        if step_idx % TARGET_UPDATE_FREQ == 0:
                            target_net.load_state_dict(q_net.state_dict())

                        if epsilon > EPSILON_END:
                            epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY

                        prev_frame = frame_next.copy()
                        prev_action_idx = action_idx.copy()
                        prev_mask = mask_next.copy()
                        prev_checkpoint_coverage = current_checkpoint_coverage

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
                            last_checkpoint_time = time.time()  
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

                        if done:
                            last_checkpoint_time = time.time()  
                            break

                    except Exception as e:
                        continue

                print(f"📊 Episode {episode + 1} - Reward: {episode_reward:.2f}, Steps: {episode_length}, ε: {epsilon:.3f}, Avg Reward/Step: {episode_reward/max(1, episode_length):.2f}")
                
                
                if (episode + 1) % CHECKPOINT_SAVE_FREQUENCY == 0:
                    checkpoint_filename = f"trackmania_dqn_checkpoint_{episode + 1}.pth"
                    if safe_save_model(q_net.state_dict(), checkpoint_filename):
                        
                        cleanup_old_checkpoints()
                    
                    
                    with cache_lock:
                        frame_cache.clear()

            print("🎉 Training completed!")
            
            
            if not safe_save_model(q_net.state_dict(), "trackmania_dqn_final_optimized.pth"):
                print("⚠️ Could not save final model due to disk space issues")

        except KeyboardInterrupt:
            print("🛑 Training interrupted by user.")
        finally:
            with crash_detection_lock:
                crash_detection_active = False
            stop_checkpoint_detection_thread()  
            stop_control_thread_func()  
            release_all_keys()
            release_all_keys()
            
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