# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# from PIL import Image

# DATA_DIR = "training_data"

# class GameDataset(Dataset):
#     def __init__(self, data_dir):
#         with open(os.path.join(data_dir, "log.json")) as f:
#             self.entries = json.load(f)
#         self.data_dir = data_dir
#         self.transform = transforms.Compose([
#             transforms.Resize((120, 160)),
#             transforms.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.entries)

#     def __getitem__(self, idx):
#         entry = self.entries[idx]
#         img = Image.open(os.path.join(self.data_dir, entry['frame']))
#         img = self.transform(img)
#         label = torch.tensor(entry['keys'], dtype=torch.float32)
#         return img, label

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
#             nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
#             nn.Flatten(),  # Add this line to flatten the output
#             # lambda x: print(x.shape),  # Add this line to print the shape
#             # nn.Flatten(),
#             nn.Linear(17024, 128),
#             nn.ReLU(),
#             nn.Linear(128, 4),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         x = self.cnn(x)
#         print(x.shape)
#         return x

# def train():
#     dataset = GameDataset(DATA_DIR)
#     loader = DataLoader(dataset, batch_size=32, shuffle=True)

#     model = SimpleCNN()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     criterion = nn.BCELoss()

#     for epoch in range(5):
#         total_loss = 0
#         for images, labels in loader:
#             preds = model(images)
#             loss = criterion(preds, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

#     torch.save(model.state_dict(), "model.pth")

# if __name__ == "__main__":
#     train()


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import cv2
# import mss
# import time
# from torchvision import transforms
# from PIL import Image
# import keyboard
# import pygetwindow as gw

# # ----- Your model -----
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
#             nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(17024, 128), nn.ReLU(),
#             nn.Linear(128, 4)
#         )

#     def forward(self, x):
#         return self.cnn(x)  # raw logits (no sigmoid)

# # ----- Utilities -----
# transform = transforms.Compose([
#     transforms.Resize((120, 160)),
#     transforms.ToTensor()
# ])

# keys_map = ['a', 'd', 'w', 's']
# threshold = 0.5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def get_frame(mss_inst, bbox):
#     img = np.array(mss_inst.grab(bbox))
#     frame = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
#     return frame

# def compute_reward(prev, curr):
#     diff = cv2.absdiff(cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY),
#                        cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY))
#     return np.sum(diff) / (diff.shape[0] * diff.shape[1])

# def sample_action(logits):
#     probs = torch.sigmoid(logits)
#     dist = torch.distributions.Bernoulli(probs)
#     actions = dist.sample()
#     return actions, dist.log_prob(actions)

# def execute_action(actions):
#     for i, key in enumerate(keys_map):
#         if actions[i] == 1:
#             keyboard.press(key)
#         else:
#             keyboard.release(key)

# # ----- Training Loop -----
# # def train_rl(episodes=10, steps_per_episode=200):
# #     model = SimpleCNN().to(device)
# #     optimizer = optim.Adam(model.parameters(), lr=1e-4)

# #     bbox = {'top': 0, 'left': 0, 'width': 1920, 'height': 1200}
# #     with mss.mss() as sct:
# #         for episode in range(episodes):
# #             log_probs = []
# #             rewards = []

# #             prev_frame = get_frame(sct, bbox)

# #             for step in range(steps_per_episode):
# #                 frame = get_frame(sct, bbox)
# #                 input_tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)

# #                 logits = model(input_tensor)[0]
# #                 actions, log_prob = sample_action(logits)

# #                 execute_action(actions.cpu().numpy())

# #                 reward = compute_reward(prev_frame, frame)
# #                 log_probs.append(log_prob.sum())  # Sum log probs across action dimensions
# #                 rewards.append(reward)

# #                 prev_frame = frame
# #                 time.sleep(1/30)

# #             # Discount and normalize rewards
# #             discounted = []
# #             gamma = 0.99
# #             r = 0
# #             for reward in reversed(rewards):
# #                 r = reward + gamma * r
# #                 discounted.insert(0, r)
# #             discounted = torch.tensor(discounted).to(device)
# #             discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)

# #             # Compute policy gradient loss
# #             loss = -torch.stack(log_probs) * discounted
# #             loss = loss.sum()

# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()

# #             print(f"Episode {episode+1}: total reward = {sum(rewards):.2f}")

# #             # Press DELETE to reset the track
# #             keyboard.press_and_release('delete')
# #             time.sleep(2)  # Wait for reset to complete (adjust as needed)

# #     torch.save(model.state_dict(), "model_rl.pth")

# import pygetwindow as gw

# def reset_track():
#     # Get the Trackmania window
#     trackmania_window = gw.getWindowsWithTitle("Trackmania")
    
#     if not trackmania_window:
#         print("Trackmania window not found.")
#         return

#     trackmania_window = trackmania_window[0]  # Use the first Trackmania window found

#     # Activate the Trackmania window (focus it)
#     trackmania_window.activate()
#     time.sleep(0.5)  # Give it some time to focus

#     # Press DELETE to reset the track
#     keyboard.press_and_release('delete')
#     print("Pressed DELETE to reset the track.")
    
#     # Wait for reset to complete (adjust the time as needed)
#     time.sleep(5)  # Wait for 5 seconds to allow the game to reset

#     print("Track reset completed.")

# # ----- RL Loop with Reset -----
# def train_rl(episodes=10, steps_per_episode=200):
#     model = SimpleCNN().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)

#     bbox = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
#     with mss.mss() as sct:
#         for episode in range(episodes):
#             log_probs = []
#             rewards = []

#             prev_frame = get_frame(sct, bbox)

#             for step in range(steps_per_episode):
#                 frame = get_frame(sct, bbox)
#                 input_tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)

#                 logits = model(input_tensor)[0]
#                 actions, log_prob = sample_action(logits)

#                 execute_action(actions.cpu().numpy())

#                 reward = compute_reward(prev_frame, frame)
#                 log_probs.append(log_prob.sum())  # Sum log probs across action dimensions
#                 rewards.append(reward)

#                 prev_frame = frame
#                 time.sleep(1/30)

#             # --- End of episode logic ---

#             # Discount and normalize rewards
#             discounted = []
#             gamma = 0.99
#             r = 0
#             for reward in reversed(rewards):
#                 r = reward + gamma * r
#                 discounted.insert(0, r)
#             discounted = torch.tensor(discounted).to(device)
#             discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)

#             # Compute policy gradient loss
#             loss = -torch.stack(log_probs) * discounted
#             loss = loss.sum()

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             print(f"Episode {episode+1}: total reward = {sum(rewards):.2f}")

#             # Reset the track at the end of each episode
#             reset_track()  # Call the function to reset the track

#     torch.save(model.state_dict(), "model_rl.pth")


# if __name__ == "__main__":
#     train_rl()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import mss
import time
from torchvision import transforms
from PIL import Image
from pynput.keyboard import Key, Controller
import pygetwindow as gw

# Initialize pynput keyboard controller
keyboard_controller = Controller()

# ----- Your model -----
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(17024, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.cnn(x)

# ----- Utilities -----
transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor()
])

keys_map = ['a', 'd', 'w', 's']
threshold = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_frame(mss_inst, bbox):
    img = np.array(mss_inst.grab(bbox))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return frame

def compute_reward(prev, curr):
    diff = cv2.absdiff(cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY),
                       cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY))
    return np.sum(diff) / (diff.shape[0] * diff.shape[1])

def sample_action(logits):
    probs = torch.sigmoid(logits)
    dist = torch.distributions.Bernoulli(probs)
    actions = dist.sample()
    return actions, dist.log_prob(actions)

def execute_action(actions):
    for i, key in enumerate(keys_map):
        if actions[i] == 1:
            keyboard_controller.press(key)
        else:
            keyboard_controller.release(key)

def reset_track():
    windows = gw.getWindowsWithTitle("Trackmania")
    if not windows:
        print("Trackmania window not found.")
        return

    win = windows[0]
    win.activate()
    time.sleep(0.5)

    # Press and release DELETE
    keyboard_controller.press(Key.delete)
    keyboard_controller.release(Key.delete)
    print("Pressed DELETE to reset the track.")
    
    time.sleep(5)
    print("Track reset completed.")

# ----- RL Loop with Reset -----
def train_rl(episodes=10, steps_per_episode=200):
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    bbox = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    with mss.mss() as sct:
        for episode in range(episodes):
            log_probs = []
            rewards = []

            prev_frame = get_frame(sct, bbox)

            for step in range(steps_per_episode):
                frame = get_frame(sct, bbox)
                input_tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)

                logits = model(input_tensor)[0]
                actions, log_prob = sample_action(logits)

                execute_action(actions.cpu().numpy())

                reward = compute_reward(prev_frame, frame)
                log_probs.append(log_prob.sum())
                rewards.append(reward)

                prev_frame = frame
                time.sleep(1 / 30)

            # --- End of episode logic ---
            discounted = []
            gamma = 0.99
            r = 0
            for reward in reversed(rewards):
                r = reward + gamma * r
                discounted.insert(0, r)
            discounted = torch.tensor(discounted).to(device)
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)

            loss = -torch.stack(log_probs) * discounted
            loss = loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Episode {episode + 1}: total reward = {sum(rewards):.2f}")

            reset_track()

    torch.save(model.state_dict(), "model_rl.pth")


if __name__ == "__main__":
    train_rl()
