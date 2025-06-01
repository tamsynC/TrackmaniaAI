import torch
import torch.nn as nn
import numpy as np
import cv2
import mss
import win32gui
from torchvision import transforms
import time
import threading
from queue import Queue
import keyboard
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from collections import deque
import os

# Import your model constants and classes
MODEL_PATH = 'trackmania_dqn_final_optimized.pth'
NUM_CLASSES = 5
INPUT_SIZE = (512, 512)

ACTIONS = ['a', 'd', 'w', 's']
NUM_ACTIONS = len(ACTIONS)

LABEL_BACKGROUND = 0
LABEL_CAR = 1
LABEL_TRACK = 4
LABEL_CHECKPOINT = 2
LABEL_FINISH = 3

# Testing parameters
TEST_EPISODES = 5
MAX_EPISODE_LENGTH = 600  # 10 minutes max per episode
CHECKPOINT_TIMEOUT = 120  # 2 minutes without checkpoint
VISUALIZATION_ENABLED = True
SAVE_STATISTICS = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Thread control for actions
action_queue = Queue()
control_thread = None
stop_control_thread = False
current_action = [0, 0, 0, 0]
action_lock = threading.Lock()

class QNetwork(nn.Module):
    """Same architecture as your training script"""
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

class TrackmaniaModelTester:
    def __init__(self, model_path, segmentation_model_path):
        """Initialize the tester with model paths"""
        self.model_path = model_path
        self.seg_model_path = segmentation_model_path
        self.transform = transforms.ToTensor()
        
        # Load models
        self.load_models()
        
        # Statistics tracking
        self.reset_statistics()
        
        # Control thread setup
        self.setup_control_thread()
        
    def load_models(self):
        """Load both the RL model and segmentation model"""
        print("🔧 Loading models...")
        
        # Load segmentation model
        self.seg_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES
        )
        self.seg_model.load_state_dict(torch.load(self.seg_model_path, map_location=device))
        self.seg_model.to(device)
        self.seg_model.eval()
        
        # Load RL model
        self.q_net = QNetwork((3, INPUT_SIZE[0], INPUT_SIZE[1]), NUM_ACTIONS).to(device)
        try:
            checkpoint = torch.load(self.model_path, map_location=device)
            self.q_net.load_state_dict(checkpoint)
            self.q_net.eval()
            print(f"✅ Successfully loaded RL model from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading RL model: {e}")
            raise
    
    def setup_control_thread(self):
        """Setup the control thread for smooth key presses"""
        global control_thread, stop_control_thread
        stop_control_thread = False
        control_thread = threading.Thread(target=self.control_thread_worker, daemon=True)
        control_thread.start()
        print("🎮 Control thread started")
    
    def control_thread_worker(self):
        """Control thread worker function"""
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
                
                for i, should_press in enumerate(action_to_execute):
                    if should_press:
                        keyboard.press(ACTIONS[i])
                    else:
                        keyboard.release(ACTIONS[i])
                
                time.sleep(0.016)  # ~60 FPS
                
            except Exception as e:
                print(f"Control thread error: {e}")
                time.sleep(0.1)
    
    def reset_statistics(self):
        """Reset all statistics for a new test session"""
        self.stats = {
            'episodes_completed': 0,
            'total_checkpoints': 0,
            'total_crashes': 0,
            'total_timeouts': 0,
            'total_finishes': 0,
            'episode_rewards': [],
            'episode_lengths': [],
            'checkpoint_times': [],
            'action_distribution': [0, 0, 0, 0],  # [right, left, forward, backward]
            'avg_track_direction': []
        }
    
    def find_trackmania_window(self):
        """Find the Trackmania window for screen capture"""
        def enum_windows_callback(hwnd, window_list):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "Trackmania" in title:
                    window_list.append(hwnd)
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        if not windows:
            raise Exception("❌ Trackmania window not found. Make sure the game is running!")
        
        hwnd = windows[0]
        rect = win32gui.GetWindowRect(hwnd)
        return {"top": rect[1], "left": rect[0], "width": rect[2] - rect[0], "height": rect[3] - rect[1]}
    
    def segment_frame(self, frame):
        """Segment frame using the segmentation model"""
        resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        input_tensor = self.transform(resized).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = self.seg_model(input_tensor)
            mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
        return mask, resized
    
    def preprocess_frame(self, frame):
        """Preprocess frame for the RL model"""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            tensor = self.transform(frame).float()
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
        return tensor.numpy()
    
    def get_centroid(self, binary_mask):
        """Get centroid of a binary mask"""
        ys, xs = np.where(binary_mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return (int(np.mean(xs)), int(np.mean(ys)))
    
    def calculate_track_direction(self, mask):
        """Calculate track direction relative to car"""
        car_mask = (mask == LABEL_CAR)
        track_mask = (mask == LABEL_TRACK)
        
        car_center = self.get_centroid(car_mask)
        if car_center is None:
            return 0
        
        track_ys, track_xs = np.where(track_mask)
        if len(track_xs) == 0:
            return 0
        
        # Sample track pixels for direction calculation
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
    
    def select_action(self, state, track_direction, use_exploration=False, epsilon=0.05):
        """Select action using the trained model"""
        if use_exploration and np.random.random() < epsilon:
            # Small amount of exploration for testing
            return [np.random.randint(0, 2) for _ in range(4)]
        
        # Use the trained policy
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
            action_probs = torch.sigmoid(q_values.squeeze())
            
            # Convert to discrete actions
            action_idx = [
                int(action_probs[0].item() > 0.5),  # Right
                int(action_probs[1].item() > 0.5),  # Left
                int(action_probs[2].item() > 0.4),  # Forward
                int(action_probs[3].item() > 0.6),  # Backward
            ]
        
        # Update action distribution statistics
        for i, action in enumerate(action_idx):
            if action:
                self.stats['action_distribution'][i] += 1
        
        return action_idx
    
    def press_action(self, action_idx):
        """Send action to the control thread"""
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
    
    def detect_events(self, mask, prev_mask=None):
        """Detect game events from the segmented mask"""
        events = []
        
        car_mask = (mask == LABEL_CAR)
        track_mask = (mask == LABEL_TRACK)
        checkpoint_mask = (mask == LABEL_CHECKPOINT)
        finish_mask = (mask == LABEL_FINISH)
        
        # Check for car on track
        if np.sum(car_mask) == 0:
            events.append("no_car_detected")
        
        # Check track coverage (crash detection)
        total_pixels = mask.shape[0] * mask.shape[1]
        track_coverage = np.sum(track_mask) / total_pixels
        if track_coverage < 0.20:
            events.append("crash")
        
        # Checkpoint detection
        checkpoint_coverage = np.sum(checkpoint_mask) / total_pixels
        if checkpoint_coverage > 0.07:  # 7% threshold
            events.append("checkpoint")
        
        # Finish line detection
        if np.any(finish_mask & car_mask):
            events.append("finish")
        
        # Stuck detection (compare with previous mask if available)
        if prev_mask is not None:
            similarity = np.mean(prev_mask == mask)
            if similarity > 0.985:
                events.append("potential_stuck")
        
        return events, track_coverage, checkpoint_coverage
    
    def restart_track(self):
        """Restart the current track"""
        print("🔄 Restarting track...")
        
        # Clear action queue
        while not action_queue.empty():
            try:
                action_queue.get_nowait()
                action_queue.task_done()
            except:
                break
        
        # Release all keys
        for key in ACTIONS:
            keyboard.release(key)
        
        time.sleep(0.2)
        
        # Press backspace to restart
        for _ in range(3):
            try:
                keyboard.press_and_release('Backspace')
                break
            except:
                time.sleep(0.1)
        
        time.sleep(2)
    
    def visualize_segmentation(self, frame, mask, action_idx, q_values=None):
        """Create a visualization of the current state"""
        if not VISUALIZATION_ENABLED:
            return
        
        # Create colored mask
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask == LABEL_CAR] = [255, 0, 0]      # Red for car
        colored_mask[mask == LABEL_TRACK] = [0, 255, 0]    # Green for track
        colored_mask[mask == LABEL_CHECKPOINT] = [0, 0, 255] # Blue for checkpoint
        colored_mask[mask == LABEL_FINISH] = [255, 255, 0]  # Yellow for finish
        
        # Overlay mask on frame
        overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
        
        # Add action text
        action_text = f"Action: R:{action_idx[0]} L:{action_idx[1]} F:{action_idx[2]} B:{action_idx[3]}"
        cv2.putText(overlay, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the frame
        cv2.imshow('Trackmania RL Test', overlay)
        cv2.waitKey(1)
    
    def run_test_episode(self, episode_num):
        """Run a single test episode"""
        print(f"\n🏁 Starting Test Episode {episode_num + 1}")
        
        episode_start_time = time.time()
        episode_length = 0
        episode_reward = 0
        checkpoints_reached = 0
        last_checkpoint_time = time.time()
        prev_mask = None
        stuck_counter = 0
        
        monitor = self.find_trackmania_window()
        print(f"🎮 Capturing window: {monitor}")
        
        with mss.mss() as sct:
            while True:
                current_time = time.time()
                
                # Check for timeout
                if current_time - episode_start_time > MAX_EPISODE_LENGTH:
                    print("⏰ Episode timeout")
                    self.stats['total_timeouts'] += 1
                    break
                
                # Check for checkpoint timeout
                if current_time - last_checkpoint_time > CHECKPOINT_TIMEOUT:
                    print("⚠️ Checkpoint timeout - restarting")
                    self.restart_track()
                    break
                
                try:
                    # Capture and process frame
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)[..., :3]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Segment frame
                    mask, resized_frame = self.segment_frame(frame)
                    state = self.preprocess_frame(resized_frame)
                    
                    # Calculate track direction
                    track_direction = self.calculate_track_direction(mask)
                    self.stats['avg_track_direction'].append(track_direction)
                    
                    # Select and execute action
                    action_idx = self.select_action(state, track_direction, use_exploration=True)
                    self.press_action(action_idx)
                    
                    # Detect events
                    events, track_coverage, checkpoint_coverage = self.detect_events(mask, prev_mask)
                    
                    # Handle events
                    if "checkpoint" in events:
                        checkpoints_reached += 1
                        last_checkpoint_time = current_time
                        self.stats['total_checkpoints'] += 1
                        self.stats['checkpoint_times'].append(current_time - episode_start_time)
                        episode_reward += 100
                        print(f"✅ Checkpoint {checkpoints_reached} reached!")
                    
                    if "finish" in events:
                        print(f"🏆 RACE COMPLETED! Time: {current_time - episode_start_time:.2f}s")
                        self.stats['total_finishes'] += 1
                        episode_reward += 1000
                        break
                    
                    if "crash" in events:
                        print(f"💥 Crash detected (track coverage: {track_coverage:.2f})")
                        self.stats['total_crashes'] += 1
                        self.restart_track()
                        break
                    
                    if "potential_stuck" in events:
                        stuck_counter += 1
                        if stuck_counter > 30:  # 30 frames of being stuck
                            print("🔄 Car appears stuck - restarting")
                            self.restart_track()
                            break
                    else:
                        stuck_counter = 0
                    
                    # Visualization
                    self.visualize_segmentation(resized_frame, mask, action_idx)
                    
                    # Small reward for staying alive
                    episode_reward += 0.1
                    
                    prev_mask = mask.copy()
                    episode_length += 1
                    
                    time.sleep(0.016)  # ~60 FPS
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    continue
        
        # Record episode statistics
        self.stats['episodes_completed'] += 1
        self.stats['episode_rewards'].append(episode_reward)
        self.stats['episode_lengths'].append(episode_length)
        
        episode_time = time.time() - episode_start_time
        print(f"📊 Episode {episode_num + 1} completed:")
        print(f"   Time: {episode_time:.2f}s")
        print(f"   Steps: {episode_length}")
        print(f"   Reward: {episode_reward:.2f}")
        print(f"   Checkpoints: {checkpoints_reached}")
    
    def run_test_session(self, num_episodes=TEST_EPISODES):
        """Run a complete test session"""
        print(f"🚀 Starting Trackmania RL Model Test Session")
        print(f"📝 Running {num_episodes} episodes")
        print(f"⚙️ Visualization: {'ON' if VISUALIZATION_ENABLED else 'OFF'}")
        print(f"💾 Statistics saving: {'ON' if SAVE_STATISTICS else 'OFF'}")
        print("\n" + "="*50)
        
        try:
            for episode in range(num_episodes):
                self.run_test_episode(episode)
                
                # Brief pause between episodes
                if episode < num_episodes - 1:
                    print("⏸️ Brief pause before next episode...")
                    time.sleep(3)
            
            # Print final statistics
            self.print_final_statistics()
            
            # Save statistics if enabled
            if SAVE_STATISTICS:
                self.save_statistics()
        
        except KeyboardInterrupt:
            print("\n🛑 Test session interrupted by user")
            self.print_final_statistics()
        
        finally:
            self.cleanup()
    
    def print_final_statistics(self):
        """Print comprehensive test statistics"""
        print("\n" + "="*50)
        print("📊 FINAL TEST STATISTICS")
        print("="*50)
        
        stats = self.stats
        
        print(f"Episodes completed: {stats['episodes_completed']}")
        print(f"Total finishes: {stats['total_finishes']}")
        print(f"Total checkpoints: {stats['total_checkpoints']}")
        print(f"Total crashes: {stats['total_crashes']}")
        print(f"Total timeouts: {stats['total_timeouts']}")
        
        if stats['episode_rewards']:
            print(f"Average episode reward: {np.mean(stats['episode_rewards']):.2f}")
            print(f"Best episode reward: {np.max(stats['episode_rewards']):.2f}")
            print(f"Average episode length: {np.mean(stats['episode_lengths']):.2f} steps")
        
        if stats['checkpoint_times']:
            print(f"Average time to first checkpoint: {np.mean(stats['checkpoint_times']):.2f}s")
            print(f"Fastest checkpoint: {np.min(stats['checkpoint_times']):.2f}s")
        
        if stats['avg_track_direction']:
            avg_direction = np.mean(np.abs(stats['avg_track_direction']))
            print(f"Average track direction deviation: {avg_direction:.3f}")
        
        # Action distribution
        total_actions = sum(stats['action_distribution'])
        if total_actions > 0:
            print("\nAction Distribution:")
            actions = ['Right', 'Left', 'Forward', 'Backward']
            for i, action in enumerate(actions):
                percentage = (stats['action_distribution'][i] / total_actions) * 100
                print(f"  {action}: {percentage:.1f}%")
        
        # Performance metrics
        if stats['episodes_completed'] > 0:
            success_rate = (stats['total_finishes'] / stats['episodes_completed']) * 100
            crash_rate = (stats['total_crashes'] / stats['episodes_completed']) * 100
            print(f"\nPerformance Metrics:")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Crash rate: {crash_rate:.1f}%")
            print(f"  Avg checkpoints per episode: {stats['total_checkpoints'] / stats['episodes_completed']:.1f}")
    
    def save_statistics(self):
        """Save statistics to file"""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trackmania_test_stats_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        stats_to_save = self.stats.copy()
        for key, value in stats_to_save.items():
            if isinstance(value, np.ndarray):
                stats_to_save[key] = value.tolist()
        
        try:
            with open(filename, 'w') as f:
                json.dump(stats_to_save, f, indent=2)
            print(f"💾 Statistics saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving statistics: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        global stop_control_thread
        
        print("🧹 Cleaning up...")
        
        # Stop control thread
        stop_control_thread = True
        if control_thread and control_thread.is_alive():
            control_thread.join(timeout=1.0)
        
        # Release all keys
        for key in ACTIONS:
            try:
                keyboard.release(key)
            except:
                pass
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        print("✅ Cleanup completed")

def main():
    """Main function to run the test"""
    # Configuration
    MODEL_PATHS = [
        "trackmania_dqn_final_optimized.pth",
        # Add more model paths here to test different checkpoints
    ]
    
    SEGMENTATION_MODEL_PATH = "unet_model.pth"
    
    print("🎮 Trackmania RL Model Tester")
    print("=" * 40)
    
    # Check for available models
    available_models = [path for path in MODEL_PATHS if os.path.exists(path)]
    
    if not available_models:
        print("❌ No trained models found!")
        print("Available models should be:")
        for path in MODEL_PATHS:
            print(f"  - {path}")
        return
    
    if not os.path.exists(SEGMENTATION_MODEL_PATH):
        print(f"❌ Segmentation model not found: {SEGMENTATION_MODEL_PATH}")
        return
    
    # Test each available model
    for model_path in available_models:
        print(f"\n🔍 Testing model: {model_path}")
        
        try:
            tester = TrackmaniaModelTester(model_path, SEGMENTATION_MODEL_PATH)
            tester.run_test_session(num_episodes=TEST_EPISODES)
            
        except Exception as e:
            print(f"❌ Error testing model {model_path}: {e}")
            continue
        
        # Ask user if they want to continue with next model
        if len(available_models) > 1 and model_path != available_models[-1]:
            response = input("\n🤔 Continue testing next model? (y/n): ")
            if response.lower() != 'y':
                break
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    # Make sure the game is ready
    print("🚨 IMPORTANT SETUP INSTRUCTIONS:")
    print("1. Start Trackmania 2020")
    print("2. Load a track you want to test on")
    print("3. Position your car at the starting line")
    print("4. Make sure the game window is visible")
    print("5. Press ENTER when ready...")
    
    input()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()