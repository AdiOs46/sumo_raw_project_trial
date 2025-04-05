from stable_baselines3 import SAC
from custom_env import MultiAgentCarEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json

# Create output directory if it doesn't exist
os.makedirs("evaluation_logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def load_training_metrics():
    """Load training metrics for comparison"""
    try:
        with open("logs/training_metrics.json", 'r') as f:
            return json.load(f)
    except:
        print("Warning: No training metrics found for comparison")
        return None

def evaluate_model(model, model_path, n_episodes=5):
    """Evaluate a trained model"""
    # Create the environment for evaluation
    raw_env = MultiAgentCarEnv()
    MAX_STEPS = raw_env.max_steps

    # Use GUI for visualization
    try:
        success = raw_env.start(gui=True, numVehicles=30, vType='human')
        if not success:
            raise RuntimeError("Could not start SUMO environment")
    except Exception as e:
        print(f"Error starting environment: {e}")
        return None

    # Wrap with monitor
    env = Monitor(raw_env, "evaluation_logs")
    
    # Initialize metrics
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'comfort': [],
        'efficiency': [],
        'safety': [],
        'traffic_flow': [],
        'lane_changes': [],
        'collisions': []
    }

    print(f"\n===== Starting Evaluation of {model_path} =====")
    print(f"Will run {n_episodes} episodes.")
    
    for episode in range(n_episodes):
        print(f"\nStarting episode {episode + 1}/{n_episodes}")
        try:
            obs, _ = env.reset(options={'gui': True, 'numVehicles': 20})
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            lane_changes = 0
            prev_lane = None
            collision = False
            
            while not (done or truncated) and steps < MAX_STEPS:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                # Update metrics
                total_reward += reward
                steps += 1
                
                # Track lane changes
                curr_lane = info.get("curr_sublane")
                if prev_lane is not None and curr_lane is not None and prev_lane != curr_lane:
                    lane_changes += 1
                prev_lane = curr_lane
                
                # Track collision
                if info.get("collision", False):
                    collision = True
                
                if steps % 100 == 0:
                    print(f"Step {steps}, Reward: {total_reward:.2f}")
            
            # Record episode metrics
            metrics['episode_rewards'].append(total_reward)
            metrics['episode_lengths'].append(steps)
            metrics['comfort'].append(np.mean([info.get("r_comf", 0) for _ in range(steps)]))
            metrics['efficiency'].append(np.mean([info.get("r_eff", 0) for _ in range(steps)]))
            metrics['safety'].append(np.mean([info.get("r_safe", 0) for _ in range(steps)]))
            metrics['traffic_flow'].append(np.mean([info.get("r_traffic", 0) for _ in range(steps)]))
            metrics['lane_changes'].append(lane_changes)
            metrics['collisions'].append(1 if collision else 0)
            
            print(f"Episode {episode + 1} complete: Reward={total_reward:.2f}, Steps={steps}, Lane Changes={lane_changes}")
            
        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
            continue
        
        time.sleep(1)  # Cleanup time
    
    return metrics

def compare_with_training(test_metrics, training_metrics):
    """Compare test metrics with training metrics"""
    if not training_metrics:
        return
    
    print("\n===== Performance Comparison =====")
    print("Training vs Testing Metrics:")
    
    for metric in ['episode_rewards', 'episode_lengths', 'comfort', 'efficiency', 'safety', 'traffic_flow']:
        train_mean = np.mean(training_metrics[metric])
        test_mean = np.mean(test_metrics[metric])
        print(f"{metric}:")
        print(f"  Training: {train_mean:.2f}")
        print(f"  Testing:  {test_mean:.2f}")
        print(f"  Difference: {((test_mean - train_mean) / train_mean * 100):.1f}%")

def plot_evaluation_results(metrics):
    """Plot evaluation results"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards and lengths
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'], marker='o')
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics['episode_lengths'], marker='o')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Plot component metrics
    plt.subplot(2, 2, 3)
    plt.plot(metrics['comfort'], label='Comfort')
    plt.plot(metrics['efficiency'], label='Efficiency')
    plt.plot(metrics['safety'], label='Safety')
    plt.plot(metrics['traffic_flow'], label='Traffic Flow')
    plt.title('Component Metrics')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot lane changes and collisions
    plt.subplot(2, 2, 4)
    plt.plot(metrics['lane_changes'], marker='o', label='Lane Changes')
    plt.plot(metrics['collisions'], marker='o', label='Collisions')
    plt.title('Lane Changes and Collisions')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    plt.close()

def main():
    # Load the trained model
    model_path = "models/sac_sumo_final"
    try:
        model = SAC.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please ensure you have a trained model before running tests.")
        return
    
    # Load training metrics for comparison
    training_metrics = load_training_metrics()
    
    # Run evaluation
    test_metrics = evaluate_model(model, model_path)
    
    if test_metrics:
        # Compare with training metrics
        compare_with_training(test_metrics, training_metrics)
        
        # Plot results
        plot_evaluation_results(test_metrics)
        
        # Print summary statistics
        print("\n===== Evaluation Summary =====")
        print(f"Average Reward: {np.mean(test_metrics['episode_rewards']):.2f}")
        print(f"Average Episode Length: {np.mean(test_metrics['episode_lengths']):.2f}")
        print(f"Average Lane Changes: {np.mean(test_metrics['lane_changes']):.2f}")
        print(f"Collision Rate: {(np.mean(test_metrics['collisions']) * 100):.1f}%")
        print("=============================")

if __name__ == "__main__":
    main()