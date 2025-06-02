import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import cv2
import mss
import win32gui
from torchvision import transforms
import time
import random
from collections import deque
import threading
from queue import Queue
import os
import keyboard
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = 'unet_model.pth'
NUM_CLASSES = 5
INPUT_SIZE = (256, 256)  # Reduced for better performance
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Action space - simplified to 4 discrete actions
ACTIONS = ['w', 'a', 's', 'd']  # Forward, Left, Back, Right
NUM_ACTIONS = len(ACTIONS)

# Segmentation labels
LABEL_BACKGROUND = 0
LABEL_CAR = 1
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3
LABEL_TRACK = 4

# Training hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
BATCH_SIZE = 64
ROLLOUT_LENGTH = 2048

# Game constants
EPISODE_TIMEOUT = 120
CHECKPOINT_REWARD = 100
FINISH_REWARD = 1000
CRASH_PENALTY = -50
STUCK_PENALTY = -10
SPEED_REWARD_SCALE = 0.1

@dataclass
class Experience:
    """Single experience tuple"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float
    advantage: float = 0.0
    return_: float = 0.0

class FeatureExtractor:
    """Extract meaningful features from segmentation mask"""
    
    def __init__(self, input_size: Tuple[int, int]):
        self.input_size = input_size
        self.prev_car_pos = None
        self.prev_time = None
        
    def extract_features(self, mask: np.ndarray) -> Dict[str, float]:
        """Extract game-relevant features from segmentation mask"""
        features = {}
        
        # Car position and presence
        car_mask = (mask == LABEL_CAR)
        car_pixels = np.sum(car_mask)
        features['car_visible'] = float(car_pixels > 0)
        
        if car_pixels > 0:
            car_center = self._get_centroid(car_mask)
            features['car_x'] = car_center[0] / self.input_size[1]  # Normalized
            features['car_y'] = car_center[1] / self.input_size[0]
            
            # Calculate speed
            current_time = time.time()
            if self.prev_car_pos is not None and self.prev_time is not None:
                dt = current_time - self.prev_time
                if dt > 0:
                    distance = np.linalg.norm(np.array(car_center) - np.array(self.prev_car_pos))
                    features['speed'] = min(distance / dt / 100.0, 1.0)  # Normalized speed
                else:
                    features['speed'] = 0.0
            else:
                features['speed'] = 0.0
                
            self.prev_car_pos = car_center
            self.prev_time = current_time
        else:
            features['car_x'] = 0.5
            features['car_y'] = 0.5
            features['speed'] = 0.0
        
        # Track coverage and direction
        track_mask = (mask == LABEL_TRACK)
        track_coverage = np.sum(track_mask) / (self.input_size[0] * self.input_size[1])
        features['track_coverage'] = track_coverage
        
        # Track direction relative to car
        if car_pixels > 0 and track_coverage > 0.1:
            car_center = self._get_centroid(car_mask)
            track_direction = self._calculate_track_direction(track_mask, car_center)
            features['track_direction'] = track_direction
        else:
            features['track_direction'] = 0.0
        
        # Checkpoint detection
        checkpoint_mask = (mask == LABEL_CHECKPOINT)
        checkpoint_coverage = np.sum(checkpoint_mask) / (self.input_size[0] * self.input_size[1])
        features['checkpoint_coverage'] = checkpoint_coverage
        features['checkpoint_visible'] = float(checkpoint_coverage > 0.001)
        features['checkpoint_close'] = float(checkpoint_coverage > 0.05)
        
        # Finish line detection
        finish_mask = (mask == LABEL_FINISH)
        finish_coverage = np.sum(finish_mask) / (self.input_size[0] * self.input_size[1])
        features['finish_coverage'] = finish_coverage
        features['finish_visible'] = float(finish_coverage > 0.001)
        
        return features
    
    def _get_centroid(self, binary_mask: np.ndarray) -> Tuple[int, int]:
        """Get centroid of binary mask"""
        ys, xs = np.where(binary_mask)
        if len(xs) == 0:
            return (self.input_size[1] // 2, self.input_size[0] // 2)
        return (int(np.mean(xs)), int(np.mean(ys)))
    
    def _calculate_track_direction(self, track_mask: np.ndarray, car_center: Tuple[int, int]) -> float:
        """Calculate track direction relative to car position"""
        track_ys, track_xs = np.where(track_mask)
        if len(track_xs) == 0:
            return 0.0
        
        car_x = car_center[0]
        left_pixels = np.sum(track_xs < car_x - 10)
        right_pixels = np.sum(track_xs > car_x + 10)
        
        total_pixels = left_pixels + right_pixels
        if total_pixels == 0:
            return 0.0
        
        # Return value between -1 (turn left) and 1 (turn right)
        return (right_pixels - left_pixels) / total_pixels

class StateProcessor:
    """Process game state into neural network input"""
    
    def __init__(self, input_size: Tuple[int, int]):
        self.input_size = input_size
        self.feature_extractor = FeatureExtractor(input_size)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def process_state(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Convert frame and mask to network input"""
        # Resize frame to input size
        frame_resized = cv2.resize(frame, self.input_size)
        
        # Convert RGB frame to tensor
        frame_tensor = self.transform(frame_resized)
        
        # Process mask into feature channels
        mask_resized = cv2.resize(mask.astype(np.uint8), self.input_size, interpolation=cv2.INTER_NEAREST)
        
        # Create separate channels for each class
        mask_channels = []
        for class_id in range(NUM_CLASSES):
            class_mask = (mask_resized == class_id).astype(np.float32)
            mask_channels.append(class_mask)
        
        mask_tensor = torch.tensor(np.stack(mask_channels), dtype=torch.float32)
        
        # Extract numerical features
        features = self.feature_extractor.extract_features(mask_resized)
        feature_tensor = torch.tensor(list(features.values()), dtype=torch.float32)
        
        # Combine all inputs
        combined_state = {
            'visual': torch.cat([frame_tensor, mask_tensor], dim=0),  # 3+5=8 channels
            'features': feature_tensor  # Numerical features
        }
        
        return combined_state

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with visual and feature processing"""
    
    def __init__(self, num_actions: int, feature_dim: int):
        super().__init__()
        
        # Visual feature extractor (CNN)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate visual feature size
        visual_feature_size = 128 * 4 * 4
        
        # Feature processor for numerical features
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined feature processor
        combined_size = visual_feature_size + 64
        self.combined_encoder = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state_dict):
        visual_input = state_dict['visual']
        feature_input = state_dict['features']
        
        # Process visual input
        visual_features = self.visual_encoder(visual_input)
        visual_features = visual_features.view(visual_features.size(0), -1)
        
        # Process numerical features
        feature_features = self.feature_encoder(feature_input)
        
        # Combine features
        combined = torch.cat([visual_features, feature_features], dim=1)
        shared_features = self.combined_encoder(combined)
        
        # Get policy and value
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        
        return action_logits, value

class GameEnvironment:
    """Trackmania game environment interface"""
    
    def __init__(self):
        self.seg_model = self._load_segmentation_model()
        self.state_processor = StateProcessor(INPUT_SIZE)
        self.monitor = self._find_trackmania_window()
        
        # Game state tracking
        self.episode_start_time = None
        self.last_checkpoint_time = None
        self.checkpoint_count = 0
        self.prev_checkpoint_coverage = 0.0
        self.stuck_frames = 0
        self.prev_car_pos = None
        
        # Control system
        self.current_keys = set()
        self.control_lock = threading.Lock()
        
        logger.info(f"Environment initialized. Window: {self.monitor}")
    
    def _load_segmentation_model(self):
        """Load the segmentation model"""
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    
    def _find_trackmania_window(self):
        """Find Trackmania window"""
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
        return {
            "top": rect[1], 
            "left": rect[0], 
            "width": rect[2] - rect[0], 
            "height": rect[3] - rect[1]
        }
    
    def reset(self):
        """Reset the environment for a new episode"""
        self._release_all_keys()
        self._restart_track()
        
        # Reset tracking variables
        self.episode_start_time = time.time()
        self.last_checkpoint_time = time.time()
        self.checkpoint_count = 0
        self.prev_checkpoint_coverage = 0.0
        self.stuck_frames = 0
        self.prev_car_pos = None
        
        # Get initial state
        time.sleep(1.0)  # Wait for restart
        return self._get_current_state()
    
    def step(self, action: int):
        """Execute action and return next state, reward, done"""
        # Execute action
        self._execute_action(action)
        
        # Small delay for action to take effect
        time.sleep(0.05)
        
        # Get next state
        next_state = self._get_current_state()
        
        # Calculate reward and check if done
        reward, done, info = self._calculate_reward_and_done(next_state)
        
        return next_state, reward, done, info
    
    def _get_current_state(self):
        """Capture and process current game state"""
        with mss.mss() as sct:
            # Capture frame
            sct_img = sct.grab(self.monitor)
            frame = np.array(sct_img)[..., :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get segmentation
            mask = self._segment_frame(frame)
            
            # Process state
            state = self.state_processor.process_state(frame, mask)
            
            return state
    
    def _segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Segment frame using the CNN model"""
        # Resize frame for segmentation
        resized = cv2.resize(frame, INPUT_SIZE)
        
        # Convert to tensor
        transform = transforms.ToTensor()
        input_tensor = transform(resized).unsqueeze(0).to(DEVICE)
        
        # Get segmentation
        with torch.no_grad():
            output = self.seg_model(input_tensor)
            mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
        return mask
    
    def _calculate_reward_and_done(self, state):
        """Calculate reward and determine if episode is done"""
        reward = 0.0
        done = False
        info = {}
        
        # Extract features from current state
        features = self.state_processor.feature_extractor.extract_features(
            cv2.resize(self._get_current_mask(), INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
        )
        
        # Check for episode timeout
        if time.time() - self.episode_start_time > EPISODE_TIMEOUT:
            done = True
            reward += -100
            info['timeout'] = True
        
        # Check for crash (no track visible)
        if features['track_coverage'] < 0.1:
            done = True
            reward += CRASH_PENALTY
            info['crash'] = True
        
        # Check for finish line
        if features['finish_coverage'] > 0.05:
            done = True
            reward += FINISH_REWARD
            info['finish'] = True
        
        # Checkpoint reward
        current_checkpoint_coverage = features['checkpoint_coverage']
        if current_checkpoint_coverage > 0.05 and self.prev_checkpoint_coverage <= 0.05:
            # Checkpoint reached
            time_bonus = max(0, 20 - (time.time() - self.last_checkpoint_time))
            reward += CHECKPOINT_REWARD + time_bonus
            self.checkpoint_count += 1
            self.last_checkpoint_time = time.time()
            info['checkpoint'] = True
        
        self.prev_checkpoint_coverage = current_checkpoint_coverage
        
        # Speed reward (encourage forward movement)
        speed_reward = features['speed'] * SPEED_REWARD_SCALE
        reward += speed_reward
        
        # Staying on track reward
        track_reward = features['track_coverage'] * 2.0
        reward += track_reward
        
        # Small penalty for time (encourage completion)
        reward -= 0.1
        
        return reward, done, info
    
    def _get_current_mask(self):
        """Get current segmentation mask (helper method)"""
        with mss.mss() as sct:
            sct_img = sct.grab(self.monitor)
            frame = np.array(sct_img)[..., :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self._segment_frame(frame)
    
    def _execute_action(self, action: int):
        """Execute the given action"""
        with self.control_lock:
            # Release all current keys
            self._release_all_keys()
            
            # Press the selected action key
            if 0 <= action < len(ACTIONS):
                key = ACTIONS[action]
                keyboard.press(key)
                self.current_keys.add(key)
    
    def _release_all_keys(self):
        """Release all currently pressed keys"""
        for key in self.current_keys:
            keyboard.release(key)
        self.current_keys.clear()
    
    def _restart_track(self):
        """Restart the current track"""
        self._release_all_keys()
        time.sleep(0.1)
        keyboard.press_and_release('backspace')
        time.sleep(0.5)

class PPORolloutBuffer:
    """Buffer for storing PPO rollout data"""
    
    def __init__(self, size: int, feature_dim: int):
        self.size = size
        self.feature_dim = feature_dim
        self.reset()
    
    def reset(self):
        """Reset the buffer"""
        self.states_visual = []
        self.states_features = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        self.ptr = 0
    
    def add(self, state, action, reward, value, log_prob, done):
        """Add experience to buffer"""
        self.states_visual.append(state['visual'].clone())
        self.states_features.append(state['features'].clone())
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.ptr += 1
    
    def compute_advantages(self, last_value: float):
        """Compute GAE advantages and returns"""
        rewards = np.array(self.rewards + [last_value])
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [False])
        
        advantages = np.zeros_like(rewards[:-1])
        advantage = 0
        
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            advantage = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * advantage
            advantages[t] = advantage
        
        returns = advantages + np.array(self.values)
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
    
    def get_batches(self, batch_size: int):
        """Get mini-batches for training"""
        indices = np.random.permutation(len(self.actions))
        
        for start_idx in range(0, len(self.actions), batch_size):
            end_idx = min(start_idx + batch_size, len(self.actions))
            batch_indices = indices[start_idx:end_idx]
            
            states_visual = torch.stack([self.states_visual[i] for i in batch_indices])
            states_features = torch.stack([self.states_features[i] for i in batch_indices])
            states = {'visual': states_visual, 'features': states_features}
            
            actions = torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.long)
            old_log_probs = torch.tensor([self.log_probs[i] for i in batch_indices])
            advantages = torch.tensor([self.advantages[i] for i in batch_indices])
            returns = torch.tensor([self.returns[i] for i in batch_indices])
            
            yield states, actions, old_log_probs, advantages, returns

class PPOAgent:
    """PPO Agent for Trackmania"""
    
    def __init__(self, feature_dim: int):
        self.network = ActorCriticNetwork(NUM_ACTIONS, feature_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.rollout_buffer = PPORolloutBuffer(ROLLOUT_LENGTH, feature_dim)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def select_action(self, state):
        """Select action using current policy"""
        with torch.no_grad():
            # Add batch dimension
            state_batch = {
                'visual': state['visual'].unsqueeze(0).to(DEVICE),
                'features': state['features'].unsqueeze(0).to(DEVICE)
            }
            
            action_logits, value = self.network(state_batch)
            
            # Sample action from policy
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def update(self):
        """Update the policy using PPO"""
        # Compute advantages
        with torch.no_grad():
            # Get last value for advantage computation
            dummy_state = {
                'visual': torch.zeros(1, 8, *INPUT_SIZE).to(DEVICE),
                'features': torch.zeros(1, self.rollout_buffer.feature_dim).to(DEVICE)
            }
            _, last_value = self.network(dummy_state)
            last_value = last_value.item()
        
        self.rollout_buffer.compute_advantages(last_value)
        
        # PPO update
        for epoch in range(PPO_EPOCHS):
            for batch in self.rollout_buffer.get_batches(BATCH_SIZE):
                states, actions, old_log_probs, advantages, returns = batch
                
                # Move to device
                states = {k: v.to(DEVICE) for k, v in states.items()}
                actions = actions.to(DEVICE)
                old_log_probs = old_log_probs.to(DEVICE)
                advantages = advantages.to(DEVICE)
                returns = returns.to(DEVICE)
                
                # Forward pass
                action_logits, values = self.network(states)
                
                # Calculate policy loss
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), returns)
                
                # Total loss
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
        
        # Reset buffer
        self.rollout_buffer.reset()
    
    def save(self, filepath: str):
        """Save the model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, filepath)
    
    def load(self, filepath: str):
        """Load the model"""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])

def main():
    """Main training loop"""
    print(f"🚀 Starting Trackmania RL Training on {DEVICE}")
    
    # Initialize environment and agent
    env = GameEnvironment()
    
    # Get feature dimension from a dummy state
    dummy_state = env._get_current_state()
    feature_dim = dummy_state['features'].shape[0]
    
    agent = PPOAgent(feature_dim)
    
    # Training loop
    episode = 0
    total_steps = 0
    
    try:
        while True:
            print(f"\n🏁 Episode {episode + 1}")
            
            # Reset environment
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Collect rollout
            for step in range(ROLLOUT_LENGTH):
                # Select action
                action, log_prob, value = agent.select_action(state)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.rollout_buffer.add(state, action, reward, value, log_prob, done)
                
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                state = next_state
                
                if done:
                    print(f"Episode ended: {info}")
                    break
            
            # Update agent
            agent.update()
            
            # Log episode results
            agent.episode_rewards.append(episode_reward)
            agent.episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
            
            # Save periodically
            if (episode + 1) % 10 == 0:
                agent.save(f"trackmania_ppo_episode_{episode + 1}.pth")
                print(f"💾 Model saved at episode {episode + 1}")
            
            # Print running statistics
            if len(agent.episode_rewards) >= 10:
                avg_reward = np.mean(agent.episode_rewards[-10:])
                avg_length = np.mean(agent.episode_lengths[-10:])
                print(f"📊 Last 10 episodes - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")
            
            episode += 1
    
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    
    finally:
        # Clean up
        env._release_all_keys()
        agent.save("trackmania_ppo_final.pth")
        
        # Plot results
        if agent.episode_rewards:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(agent.episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            plt.subplot(1, 2, 2)
            plt.plot(agent.episode_lengths)
            plt.title('Episode Lengths')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            
            plt.tight_layout()
            plt.show()
        
        print("🎉 Training completed!")

if __name__ == "__main__":
    main()