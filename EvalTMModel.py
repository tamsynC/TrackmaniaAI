# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import numpy as np
# import cv2
# import mss
# import win32gui
# from torchvision import transforms
# import time
# import random
# from collections import deque

# import keyboard

# # ===== CONFIGURATION =====
# MODEL_PATH = 'unet_model.pth'
# NUM_CLASSES = 5
# INPUT_SIZE = (512, 512)  # Resize input to this resolution

# ACTIONS = ['a', 'd', 'w', 's']  # discrete actions
# NUM_ACTIONS = len(ACTIONS)

# LABEL_BACKGROUND = 0
# LABEL_CAR = 1
# LABEL_TRACK = 4
# LABEL_CHECKPOINT = 2
# LABEL_FINISH = 3

# # Hyperparameters
# BATCH_SIZE = 32
# GAMMA = 0.99
# LEARNING_RATE = 1e-4
# REPLAY_BUFFER_CAPACITY = 10000
# TARGET_UPDATE_FREQ = 1000  # steps
# EPSILON_START = 1.0
# EPSILON_END = 0.05
# EPSILON_DECAY = 10000  # steps

# # ===== MODEL LOADING (Segmentation) =====
# import segmentation_models_pytorch as smp
# def load_segmentation_model(path):
#     model = smp.Unet(
#         encoder_name="resnet34",
#         encoder_weights=None,
#         in_channels=3,
#         classes=NUM_CLASSES
#     )
#     model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
#     model.eval()
#     return model

# # ===== ENVIRONMENT HELPERS =====

# def find_trackmania_window():
#     def enum_windows_callback(hwnd, window_list):
#         if win32gui.IsWindowVisible(hwnd):
#             title = win32gui.GetWindowText(hwnd)
#             if "Trackmania" in title:
#                 window_list.append(hwnd)
#     windows = []
#     win32gui.EnumWindows(enum_windows_callback, windows)
#     if not windows:
#         raise Exception("Trackmania window not found.")
#     hwnd = windows[0]
#     rect = win32gui.GetWindowRect(hwnd)
#     return {"top": rect[1], "left": rect[0], "width": rect[2] - rect[0], "height": rect[3] - rect[1]}

# transform = transforms.ToTensor()

# def segment_frame(model, frame):
#     resized = cv2.resize(frame, INPUT_SIZE)
#     input_tensor = transform(resized).unsqueeze(0)
#     with torch.no_grad():
#         output = model(input_tensor)
#         mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
#     return mask, resized

# # ===== MASK UTILITIES =====

# def get_bbox(binary_mask):
#     ys, xs = np.where(binary_mask)
#     if len(xs) == 0 or len(ys) == 0:
#         return None
#     return (min(xs), min(ys), max(xs), max(ys))

# def boxes_overlap(a, b):
#     if a is None or b is None:
#         return False
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

# def detect_events(mask):
#     car_mask = (mask == LABEL_CAR)
#     track_mask = (mask == LABEL_TRACK)
#     checkpoint_mask = (mask == LABEL_CHECKPOINT)

#     car_bbox = get_bbox(car_mask)
#     track_bbox = get_bbox(track_mask)

#     events = []

#     if not boxes_overlap(car_bbox, track_bbox):
#         events.append("out_of_bounds")

#     checkpoint_mask_uint8 = checkpoint_mask.astype(np.uint8)
#     num_labels, labels_im = cv2.connectedComponents(checkpoint_mask_uint8)

#     checkpoint_crossed = False
#     for label in range(1, num_labels):  # skip background label 0
#         component_mask = (labels_im == label)
#         bbox = get_bbox(component_mask)
#         if boxes_overlap(car_bbox, bbox):
#             checkpoint_crossed = True

#     if checkpoint_crossed:
#         events.append("checkpoint")

#     return events

# def reward_from_events(events):
#     reward = 0
#     if "checkpoint" in events:
#         reward += 10
#     if "out_of_bounds" in events:
#         reward -= 5
#     return reward

# # ===== Q-NETWORK =====

# class QNetwork(nn.Module):
#     def __init__(self, input_shape, num_actions):
#         super(QNetwork, self).__init__()
#         c, h, w = input_shape
#         self.conv = nn.Sequential(
#             nn.Conv2d(c, 16, kernel_size=8, stride=4), nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(),
#         )
#         # Compute conv output size
#         def conv2d_size_out(size, kernel_size=3, stride=1):
#             return (size - (kernel_size - 1) - 1) // stride + 1

#         convw = conv2d_size_out(
#             conv2d_size_out(
#                 conv2d_size_out(w, 8, 4),
#                 4, 2),
#             3, 1)

#         convh = conv2d_size_out(
#             conv2d_size_out(
#                 conv2d_size_out(h, 8, 4),
#                 4, 2),
#             3, 1)

#         linear_input_size = convw * convh * 64
#         self.fc = nn.Sequential(
#             nn.Linear(linear_input_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_actions)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)

# # ===== REPLAY BUFFER =====

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), 
#                 np.array(next_states), np.array(dones, dtype=np.uint8))

#     def __len__(self):
#         return len(self.buffer)

# # ===== ACTION FUNCTIONS =====

# def press_action(action_idx):
#     for i, key in enumerate(ACTIONS):
#         if i == action_idx:
#             keyboard.press(key)
#         else:
#             keyboard.release(key)

# def release_all_keys():
#     for key in ACTIONS:
#         keyboard.release(key)

# # ===== STATE PROCESSING =====

# def preprocess_frame(frame):
#     # Input: RGB frame (512x512x3) uint8
#     # Output: Float tensor (3,512,512), normalized 0-1
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     tensor = transform(frame).float()
#     return tensor.numpy()

# # ===== TRAINING FUNCTION =====

# def train_step(q_net, target_net, optimizer, replay_buffer):
#     if len(replay_buffer) < BATCH_SIZE:
#         return

#     states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

#     states_v = torch.FloatTensor(states)
#     next_states_v = torch.FloatTensor(next_states)
#     actions_v = torch.LongTensor(actions).unsqueeze(1)
#     rewards_v = torch.FloatTensor(rewards)
#     done_mask = torch.BoolTensor(dones)

#     state_action_values = q_net(states_v).gather(1, actions_v).squeeze()
#     next_state_values = target_net(next_states_v).max(1)[0].detach()
#     next_state_values[done_mask] = 0.0
#     expected_state_action_values = rewards_v + GAMMA * next_state_values

#     loss = F.mse_loss(state_action_values, expected_state_action_values)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# # ===== MAIN LOOP =====

# def main():
#     print("🔧 Loading segmentation model...")
#     seg_model = load_segmentation_model(MODEL_PATH)

#     q_net = QNetwork((3, INPUT_SIZE[1], INPUT_SIZE[0]), NUM_ACTIONS)
#     target_net = QNetwork((3, INPUT_SIZE[1], INPUT_SIZE[0]), NUM_ACTIONS)
#     target_net.load_state_dict(q_net.state_dict())
#     target_net.eval()

#     optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
#     replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

#     epsilon = EPSILON_START
#     epsilon_decay_step = (EPSILON_START - EPSILON_END) / EPSILON_DECAY
#     step_idx = 0

#     monitor = find_trackmania_window()
#     print(f"🎮 Capturing Trackmania window at: {monitor}")

#     cumulative_reward = 0
#     episode_reward = 0
#     episode_length = 0
#     max_episode_length = 300  # ~10 seconds at 30fps was 300
#     with mss.mss() as sct:
#         try:
#             while True:
#                 sct_img = sct.grab(monitor)
#                 frame = np.array(sct_img)[..., :3]
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#                 mask, resized_frame = segment_frame(seg_model, frame)

#                 # State for DQN: normalized RGB frame resized
#                 state = preprocess_frame(resized_frame)

#                 # Select action
#                 if random.random() < epsilon:
#                     action_idx = random.randint(0, NUM_ACTIONS - 1)
#                 else:
#                     state_t = torch.FloatTensor(state).unsqueeze(0)
#                     with torch.no_grad():
#                         q_values = q_net(state_t)
#                         action_idx = q_values.argmax(1).item()

#                 press_action(action_idx)

#                 # Take next frame and get next state after action delay
#                 time.sleep(1 / 30)  # wait ~1 frame

#                 sct_img_next = sct.grab(monitor)
#                 frame_next = np.array(sct_img_next)[..., :3]
#                 frame_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2RGB)

#                 mask_next, resized_frame_next = segment_frame(seg_model, frame_next)
#                 next_state = preprocess_frame(resized_frame_next)

#                 # Reward and done signal
#                 events = detect_events(mask_next)
#                 reward = reward_from_events(events)
#                 done = episode_length >= max_episode_length

#                 episode_reward += reward

#                 replay_buffer.push(state, action_idx, reward, next_state, done)

#                 train_step(q_net, target_net, optimizer, replay_buffer)

#                 # Target network update
#                 if step_idx % TARGET_UPDATE_FREQ == 0:
#                     target_net.load_state_dict(q_net.state_dict())

#                 # Decay epsilon
#                 if epsilon > EPSILON_END:
#                     epsilon -= epsilon_decay_step

#                 # # Visual Debug Overlay (optional)
#                 # overlay = resized_frame.copy()
#                 # for event in events:
#                 #     if event == "out_of_bounds":
#                 #         cv2.putText(overlay, "OUT OF BOUNDS", (10, 30),
#                 #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 #     if event == "checkpoint":
#                 #         cv2.putText(overlay, "CHECKPOINT", (10, 60),
#                 #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 # cv2.imshow("Trackmania DQN", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

#                 # if cv2.waitKey(1) & 0xFF == ord('q'):
#                 #     break

#                 step_idx += 1
#                 episode_length += 1

#                 if done:
#                     print(f"Episode done. Total reward: {episode_reward:.2f}")
#                     episode_length = 0
#                     episode_reward = 0
#                     release_all_keys()

#         except KeyboardInterrupt:
#             print("🛑 Training interrupted by user.")
#         finally:
#             release_all_keys()
#             cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

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

ACTIONS = ['a', 'd', 'w', 's']
ACTION_INDEX = [0, 1, 2, 3]
ACTION_WEIGHTS = [0.15, 0.15, 0.9, 0.025]
ACTION_WEIGHTS_NORMALISED = [0.125, 0.125, 0.7, 0.05]
NUM_ACTIONS = len(ACTIONS)

LABEL_BACKGROUND = 0
LABEL_CAR = 1
LABEL_TRACK = 4
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4
REPLAY_BUFFER_CAPACITY = 10000
TARGET_UPDATE_FREQ = 1000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 5000

# ===== LOAD SEGMENTATION MODEL =====
def load_segmentation_model(path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
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
    input_tensor = transform(resized).unsqueeze(0)
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

    car_bbox = get_bbox(car_mask)
    track_bbox = get_bbox(track_mask)

    events = []
    if not boxes_overlap(car_bbox, track_bbox):
        events.append("out_of_bounds")

    checkpoint_mask_uint8 = checkpoint_mask.astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(checkpoint_mask_uint8)

    for label in range(1, num_labels):
        component_mask = (labels_im == label)
        bbox = get_bbox(component_mask)
        if boxes_overlap(car_bbox, bbox):
            events.append("checkpoint")
            break

    return events

def reward_from_events(events, episode_length, max_episode_length):
    reward = 0
    if "checkpoint" in events:
        reward += int(100 * round(1- float(episode_length/max_episode_length), 2))
    if "out_of_bounds" in events:
        reward -= 500
    return reward

# ===== Q-NETWORK =====
class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
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

# ===== ACTION CONTROL =====
def press_action(action_idx):

    for i in range(len(action_idx)):
        if action_idx[i] == 1:
            keyboard.press(ACTIONS[i])
            #print("pressed:", key)
        else:
            keyboard.release(ACTIONS[i])

def release_all_keys():
    for key in ACTIONS:
        keyboard.release(key)

# ===== STATE PROCESSING =====
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    tensor = transform(frame).float()
    return tensor.numpy()

# ===== TRAINING =====
def train_step(q_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states_v = torch.FloatTensor(states)
    next_states_v = torch.FloatTensor(next_states)
    actions_v = torch.LongTensor(actions).unsqueeze(1)
    rewards_v = torch.FloatTensor(rewards)
    done_mask = torch.BoolTensor(dones)
    dones_v = torch.FloatTensor(dones)

    q_values = q_net(states_v)

    # Create target tensor
    with torch.no_grad():
        next_q_values = target_net(next_states_v)
        next_q_max = next_q_values.max(1)[0]
        expected_q_values = rewards_v + (GAMMA * next_q_max * (1 - dones_v))

    actions_v = torch.FloatTensor(actions)  # shape: (batch_size, 4)
    state_action_values = (q_values * actions_v).sum(1)

    loss = F.mse_loss(state_action_values, expected_q_values)
    next_state_values = target_net(next_states_v).max(1)[0].detach()
    next_state_values[done_mask] = 0.0
    expected_state_action_values = rewards_v + GAMMA * next_state_values

    #loss = F.mse_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ===== MAIN =====
def main():
    print("🔧 Loading segmentation model...")
    seg_model = load_segmentation_model(MODEL_PATH)

    q_net = QNetwork((3, INPUT_SIZE[1], INPUT_SIZE[0]), NUM_ACTIONS)
    q_net.load_state_dict(torch.load("trackmania_dqn_final_nobrakes.pth", map_location=torch.device('cuda')))
    q_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    epsilon = EPSILON_START
    epsilon_decay_step = (EPSILON_START - EPSILON_END) / EPSILON_DECAY
    step_idx = 0

    monitor = find_trackmania_window()
    print(f"🎮 Capturing Trackmania window at: {monitor}")

    max_episode_length = 100
    total_episodes = 60

    with mss.mss() as sct:
        try:
            for episode in range(10):  # evaluate 10 times
                total_reward = 0
                episode_length = 0
                print(f"\n🎮 Evaluating Episode {episode + 1}/10")

                while episode_length < max_episode_length:
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)[..., :3]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    mask, resized_frame = segment_frame(seg_model, frame)
                    state = preprocess_frame(resized_frame)

                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = q_net(state_t)
                        probs = torch.sigmoid(q_values.squeeze())
                        action_idx = [1 if p > 0.5 else 0 for p in probs.tolist()]

                    if action_idx[3] == 1:
                        action_idx = [action_idx[0], action_idx[1], 0, action_idx[3]]

                    press_action(action_idx)
                    time.sleep(1 / 30)

                    # Observe next state and reward
                    sct_img_next = sct.grab(monitor)
                    frame_next = np.array(sct_img_next)[..., :3]
                    frame_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2RGB)
                    mask_next, resized_frame_next = segment_frame(seg_model, frame_next)

                    events = detect_events(mask_next)
                    reward = reward_from_events(events, episode_length, max_episode_length)
                    done = episode_length + 1 >= max_episode_length

                    total_reward += reward
                    episode_length += 1

                    if done:
                        print(f"✅ Evaluation Episode {episode+1} finished. Total reward: {total_reward}")
                        keyboard.press_and_release('escape')
                        time.sleep(0.1)
                        keyboard.press_and_release('enter')
                        time.sleep(0.1)
                        keyboard.press_and_release('m')
                        release_all_keys()
                        break
        except KeyboardInterrupt:
            print("🛑 Evaluation interrupted.")
        finally:
            release_all_keys()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
