import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import cv2
import mss
import win32gui
from torchvision import transforms
import time
import random
from collections import deque

import keyboard
import segmentation_models_pytorch as smp

# ===== CONFIGURATION =====
MODEL_PATH = 'unet_model.pth'
NUM_CLASSES = 5
INPUT_SIZE = (512, 512)

ACTIONS = ['a', 'd', 'w', 's']  # left, right, forward, backward
NUM_ACTIONS = len(ACTIONS)

LABEL_BACKGROUND = 0
LABEL_CAR = 1
LABEL_TRACK = 4
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3

# Simple hyperparameters that work
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4
REPLAY_BUFFER_CAPACITY = 10000
TARGET_UPDATE_FREQ = 1000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 15000

# Timeout settings
EPISODE_TIMEOUT = 300  # 5 minutes in seconds
ACTION_DURATION = 0.1  # How long to hold each action

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== LOAD SEGMENTATION MODEL =====
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

# ===== ENVIRONMENT HELPERS =====
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

def segment_frame(model, frame):
    resized = cv2.resize(frame, INPUT_SIZE)
    input_tensor = transform(resized).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return mask, resized

# ===== MASK UTILITIES =====
def get_bbox(binary_mask):
    ys, xs = np.where(binary_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (min(xs), min(ys), max(xs), max(ys))

def boxes_overlap(a, b):
    if a is None or b is None:
        return False
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

def detect_events(mask):
    car_mask = (mask == LABEL_CAR)
    track_mask = (mask == LABEL_TRACK)
    checkpoint_mask = (mask == LABEL_CHECKPOINT)
    finish_mask = (mask == LABEL_FINISH)

    car_bbox = get_bbox(car_mask)
    track_bbox = get_bbox(track_mask)

    events = []
    
    # Check if finished
    if car_bbox and np.any(finish_mask):
        finish_bbox = get_bbox(finish_mask)
        if boxes_overlap(car_bbox, finish_bbox):
            events.append("finish")
            return events  # Prioritize finish
    
    # Check for out of bounds
    if not boxes_overlap(car_bbox, track_bbox):
        events.append("out_of_bounds")

    # Check for checkpoint
    if np.any(checkpoint_mask) and car_bbox:
        checkpoint_mask_uint8 = checkpoint_mask.astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(checkpoint_mask_uint8)

        for label in range(1, num_labels):
            component_mask = (labels_im == label)
            bbox = get_bbox(component_mask)
            if boxes_overlap(car_bbox, bbox):
                events.append("checkpoint")
                break

    return events

def reward_from_events(events):
    reward = 0.1  # Small reward for staying alive
    if "finish" in events:
        reward += 100
    if "checkpoint" in events:
        reward += 10
    if "out_of_bounds" in events:
        reward -= 10
    return reward

# ===== SIMPLE Q-NETWORK =====
class SimpleQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(SimpleQNetwork, self).__init__()
        c, h, w = input_shape
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        
        # Calculate conv output size
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)

        linear_input_size = convw * convh * 64
        
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ===== REPLAY BUFFER =====
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

# ===== SIMPLE ACTION CONTROL =====
def execute_action(action_idx):
    """Execute action and hold it for ACTION_DURATION"""
    action_key = ACTIONS[action_idx]
    print(f"Executing action: {action_key}")
    
    # Release all keys first
    for key in ACTIONS:
        keyboard.release(key)
    
    # Press the selected key
    keyboard.press(action_key)
    time.sleep(ACTION_DURATION)
    keyboard.release(action_key)

def release_all_keys():
    for key in ACTIONS:
        keyboard.release(key)

def restart_race():
    """Restart the current race"""
    print("🔄 Restarting race...")
    release_all_keys()
    time.sleep(0.2)
    keyboard.press_and_release('del')  # Reset position
    time.sleep(1.0)

# ===== STATE PROCESSING =====
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    tensor = transform(frame).float()
    return tensor.numpy()

# ===== TRAINING =====
def train_step(q_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states_v = torch.FloatTensor(states).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    actions_v = torch.LongTensor(actions).to(device).unsqueeze(1)
    rewards_v = torch.FloatTensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = q_net(states_v).gather(1, actions_v).squeeze()
    next_state_values = target_net(next_states_v).max(1)[0].detach()
    next_state_values[done_mask] = 0.0
    expected_state_action_values = rewards_v + GAMMA * next_state_values

    loss = F.mse_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# ===== MAIN =====
def main():
    print("🔧 Loading segmentation model...")
    seg_model = load_segmentation_model(MODEL_PATH)

    q_net = SimpleQNetwork((3, INPUT_SIZE[1], INPUT_SIZE[0]), NUM_ACTIONS).to(device)
    target_net = SimpleQNetwork((3, INPUT_SIZE[1], INPUT_SIZE[0]), NUM_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    epsilon = EPSILON_START
    epsilon_decay_step = (EPSILON_START - EPSILON_END) / EPSILON_DECAY
    step_idx = 0

    monitor = find_trackmania_window()
    print(f"🎮 Capturing Trackmania window at: {monitor}")

    total_episodes = 100
    
    print("\n🚨 IMPORTANT: Make sure Trackmania is the active window!")
    print("Starting in 3 seconds...")
    time.sleep(3)

    with mss.mss() as sct:
        try:
            for episode in range(total_episodes):
                episode_reward = 0
                episode_start_time = time.time()
                steps_this_episode = 0
                max_steps_per_episode = 500

                print(f"\n🚗 Starting Episode {episode + 1}/{total_episodes} (ε={epsilon:.3f})")

                while steps_this_episode < max_steps_per_episode:
                    current_time = time.time()
                    episode_duration = current_time - episode_start_time
                    
                    # Check timeout
                    if episode_duration > EPISODE_TIMEOUT:
                        print(f"⏰ Episode timeout ({EPISODE_TIMEOUT}s reached)")
                        break

                    # Capture frame
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)[..., :3]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    mask, resized_frame = segment_frame(seg_model, frame)
                    state = preprocess_frame(resized_frame)

                    # Choose action
                    if random.random() < epsilon:
                        if random.random() < 0.6:  # Bias towards forward movement
                            action_idx = 2  # 'w' key (forward)
                        else:
                            action_idx = random.randint(0, NUM_ACTIONS - 1)
                    else:
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        with torch.no_grad():
                            q_values = q_net(state_t)
                            action_idx = q_values.argmax(1).item()

                    # Execute action
                    execute_action(action_idx)

                    # Get next state
                    sct_img_next = sct.grab(monitor)
                    frame_next = np.array(sct_img_next)[..., :3]
                    frame_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2RGB)

                    mask_next, resized_frame_next = segment_frame(seg_model, frame_next)
                    next_state = preprocess_frame(resized_frame_next)

                    # Calculate reward
                    events = detect_events(mask_next)
                    reward = reward_from_events(events)
                    done = "finish" in events or "out_of_bounds" in events

                    episode_reward += reward
                    replay_buffer.push(state, action_idx, reward, next_state, done)

                    # Train
                    loss = train_step(q_net, target_net, optimizer, replay_buffer)

                    # Update target network
                    if step_idx % TARGET_UPDATE_FREQ == 0:
                        target_net.load_state_dict(q_net.state_dict())
                        print(f"🎯 Target network updated at step {step_idx}")

                    # Update epsilon
                    if epsilon > EPSILON_END:
                        epsilon -= epsilon_decay_step

                    step_idx += 1
                    steps_this_episode += 1

                    # Print progress occasionally
                    if steps_this_episode % 100 == 0:
                        print(f"Step {steps_this_episode}, Reward: {episode_reward:.2f}, Action: {ACTIONS[action_idx]}")

                    if done:
                        if "finish" in events:
                            print(f"🏁 FINISHED! Episode {episode + 1} completed! Reward: {episode_reward:.2f}")
                        else:
                            print(f"❌ Episode {episode + 1} ended. Reward: {episode_reward:.2f}")
                        break

                # Restart race for next episode
                restart_race()
                
                print(f"📊 Episode {episode + 1} completed in {steps_this_episode} steps. Total reward: {episode_reward:.2f}")

            print("🎉 Training complete!")
            torch.save(q_net.state_dict(), "trackmania_dqn_simple.pth")
            print("💾 Q-network saved to trackmania_dqn_simple.pth")

        except KeyboardInterrupt:
            print("🛑 Training interrupted by user.")
        finally:
            release_all_keys()

if __name__ == "__main__":
    main()