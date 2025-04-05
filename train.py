import os
import matplotlib.pyplot as plt
import numpy as np
import traci
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
from custom_env import MultiAgentCarEnv

# Create folders for outputs
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# Custom callback for collecting metrics
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Episode-level metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Components of reward
        self.episode_comfort = []
        self.episode_efficiency = []
        self.episode_safety = []
        self.episode_lane_changes = []
        self.episode_traffic_flow = []
        self.current_episode_comfort = []
        self.current_episode_efficiency = []
        self.current_episode_safety = []
        self.current_episode_lane_changes = []
        self.current_episode_traffic_flow = []
        
        # For tracking progress
        self.n_episodes = 0
        self.lane_change_count = 0
        self.prev_lane = None
        
    def _on_step(self):
        # Track episode progress
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # Extract reward components from info
        info = self.locals["infos"][0]
        comf = info.get("r_comf", 0.0)
        eff = info.get("r_eff", 0.0)
        safe = info.get("r_safe", 0.0)
        traffic = info.get("r_traffic", 0.0)
        
        # Track lane changes
        curr_lane = info.get("curr_sublane", None)
        if self.prev_lane is not None and curr_lane is not None and self.prev_lane != curr_lane:
            self.lane_change_count += 1
            print(f"Lane change detected: {self.prev_lane} -> {curr_lane}")
        self.prev_lane = curr_lane
        
        # Debug print to verify reward components
        if self.current_episode_length % 50 == 0:
            print(f"Step {self.current_episode_length}: comf={comf:.4f}, eff={eff:.4f}, safe={safe:.4f}, traffic={traffic:.4f}")
            print(f"Lane changes so far: {self.lane_change_count}")
        
        # Log components
        self.current_episode_comfort.append(comf)
        self.current_episode_efficiency.append(eff)
        self.current_episode_safety.append(safe)
        self.current_episode_traffic_flow.append(traffic)
        
        # Check if episode ended naturally
        episode_ended = self.locals["dones"][0] or self.locals.get("truncated", [False])[0]
        
        # On episode end
        if episode_ended:
            self.n_episodes += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Store component histories
            self.episode_comfort.append(self.current_episode_comfort.copy())
            self.episode_efficiency.append(self.current_episode_efficiency.copy())
            self.episode_safety.append(self.current_episode_safety.copy())
            self.episode_traffic_flow.append(self.current_episode_traffic_flow.copy())
            self.episode_lane_changes.append(self.lane_change_count)
            
            # Print episode summary
            print(f"\nEpisode {self.n_episodes}: Reward={self.current_episode_reward:.2f}, Length={self.current_episode_length}, Lane Changes={self.lane_change_count}")
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_episode_comfort = []
            self.current_episode_efficiency = []
            self.current_episode_safety = []
            self.current_episode_traffic_flow = []
            self.lane_change_count = 0
            self.prev_lane = None
            
            # Create intermediate plots every episode
            if self.n_episodes % 1 == 0:
                self.create_intermediate_plots()
                
            # After 5 episodes, also create detailed per-component plots
            if self.n_episodes % 5 == 0:
                self.create_component_plots()
                
        return True
    
    def create_intermediate_plots(self):
        """Create and save plots during training"""
        # Plot episode rewards and lengths
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards, marker='o')
        plt.title(f'Training Rewards (Ep {self.n_episodes})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_lengths, marker='o')
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_lane_changes, marker='o', color='green')
        plt.title('Lane Changes per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Lane Changes')
        plt.grid(True)
        
        if self.episode_traffic_flow:
            plt.subplot(2, 2, 4)
            avg_traffic = [np.mean(tf) for tf in self.episode_traffic_flow]
            plt.plot(avg_traffic, marker='o', color='purple')
            plt.title('Avg Traffic Flow Reward')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("graphs", f"training_progress_ep{self.n_episodes}.png"))
        plt.close()
    
    def create_component_plots(self):
        """Create more detailed component plots"""
        if not self.episode_comfort:
            return
            
        plt.figure(figsize=(15, 12))
        
        episodes = range(1, self.n_episodes + 1)
        
        plt.subplot(4, 1, 1)
        avg_comfort = [np.mean(comf) for comf in self.episode_comfort]
        plt.plot(episodes, avg_comfort, marker='o')
        plt.title("Average Comfort Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Comfort Reward")
        plt.grid(True)
        
        plt.subplot(4, 1, 2)
        avg_efficiency = [np.mean(eff) for eff in self.episode_efficiency]
        plt.plot(episodes, avg_efficiency, marker='o')
        plt.title("Average Efficiency Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Efficiency Reward")
        plt.grid(True)
        
        plt.subplot(4, 1, 3)
        avg_safety = [np.mean(safe) for safe in self.episode_safety]
        plt.plot(episodes, avg_safety, marker='o')
        plt.title("Average Safety Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Safety Reward")
        plt.grid(True)
        
        plt.subplot(4, 1, 4)
        if self.episode_traffic_flow:
            avg_traffic = [np.mean(tf) for tf in self.episode_traffic_flow]
            plt.plot(episodes, avg_traffic, marker='o', color='purple')
            plt.title("Average Traffic Flow Per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Traffic Flow Reward")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("graphs", f"components_per_episode_{self.n_episodes}.png"))
        plt.close()
        
        plt.figure(figsize=(15, 12))
        
        last_idx = -1
        timesteps = range(len(self.episode_comfort[last_idx]))
        
        plt.subplot(4, 1, 1)
        plt.plot(timesteps, self.episode_comfort[last_idx])
        plt.title(f"Episode {self.n_episodes} - Comfort Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Comfort Reward")
        plt.grid(True)
        
        plt.subplot(4, 1, 2)
        plt.plot(timesteps, self.episode_efficiency[last_idx])
        plt.title(f"Episode {self.n_episodes} - Efficiency Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Efficiency Reward")
        plt.grid(True)
        
        plt.subplot(4, 1, 3)
        plt.plot(timesteps, self.episode_safety[last_idx])
        plt.title(f"Episode {self.n_episodes} - Safety Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Safety Reward")
        plt.grid(True)
        
        plt.subplot(4, 1, 4)
        if self.episode_traffic_flow and self.episode_traffic_flow[last_idx]:
            plt.plot(timesteps, self.episode_traffic_flow[last_idx], color='purple')
            plt.title(f"Episode {self.n_episodes} - Traffic Flow Over Time")
            plt.xlabel("Timestep")
            plt.ylabel("Traffic Flow Reward")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("graphs", f"episode_{self.n_episodes}_detail.png"))
        plt.close()

# Constants
MODEL_DIR = "models"
LOG_DIR = "logs"
TOTAL_TIMESTEPS = 100000
CHECKPOINT_FREQ = 10000

def make_env():
    """Create and configure the environment"""
    try:
        env = MultiAgentCarEnv()
        
        # Initialize the observation and action spaces if not already defined
        if not hasattr(env, 'observation_space'):
            print("Warning: Environment missing observation_space, adding default")
            env.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)
            
        if not hasattr(env, 'action_space'):
            print("Warning: Environment missing action_space, adding default")
            env.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Start the environment with randomized number of vehicles
        print("Starting environment...")
        num_vehicles = np.random.randint(20, 40)
        env.start(gui=True, numVehicles=num_vehicles, vType='human')
        
        # Modify vehicle behavior to create diverse traffic
        try:
            for i in range(num_vehicles):
                veh_name = f'vehicle_{i}'
                if veh_name in traci.vehicle.getIDList():
                    behavior_type = np.random.choice(['aggressive', 'normal', 'cautious'])
                    if behavior_type == 'aggressive':
                        traci.vehicle.setLaneChangeMode(veh_name, 259)
                        traci.vehicle.setSpeedFactor(veh_name, 1.2)
                    elif behavior_type == 'cautious':
                        traci.vehicle.setLaneChangeMode(veh_name, 512)
                        traci.vehicle.setSpeedFactor(veh_name, 0.8)
                    else:
                        traci.vehicle.setLaneChangeMode(veh_name, 256)
                        traci.vehicle.setSpeedFactor(veh_name, 1.0)
        except Exception as e:
            print(f"Warning: Could not set vehicle behaviors: {e}")
        
        # Wrap with Monitor for additional logging
        env = Monitor(env, LOG_DIR)
        return env
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise

try:
    # Create and wrap environment
    print("Creating environment...")
    env = DummyVecEnv([make_env])
    
    # Create SAC model instead of PPO
    print("Creating SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        policy_kwargs=dict(
            net_arch=[256, 256]
        )
    )
    
    # Setup callbacks
    metrics_callback = MetricsCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_DIR,
        name_prefix="sac_sumo",
    )
    
    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, metrics_callback],
    )
    
    # Save the final model
    model.save(os.path.join(MODEL_DIR, "sac_sumo_final"))
    print("Training completed. Model saved.")
    
    # PLOTTING FINAL RESULTS
    # 1) Plot reward components per episode
    if metrics_callback.episode_comfort:
        print(f"Collected data for {len(metrics_callback.episode_comfort)} episodes")
        print(f"First episode comfort data length: {len(metrics_callback.episode_comfort[0])}")
        
        plt.figure(figsize=(15, 12))
        
        # Plot average comfort per episode
        plt.subplot(4, 1, 1)
        avg_comfort = [np.mean(comf) if comf else 0 for comf in metrics_callback.episode_comfort]
        plt.plot(avg_comfort, marker='o')
        plt.title("Average Comfort Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Comfort Reward")
        plt.grid(True)
        
        # Plot average efficiency per episode
        plt.subplot(4, 1, 2)
        avg_efficiency = [np.mean(eff) if eff else 0 for eff in metrics_callback.episode_efficiency]
        plt.plot(avg_efficiency, marker='o')
        plt.title("Average Efficiency Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Efficiency Reward")
        plt.grid(True)
        
        # Plot average safety per episode
        plt.subplot(4, 1, 3)
        avg_safety = [np.mean(safe) if safe else 0 for safe in metrics_callback.episode_safety]
        plt.plot(avg_safety, marker='o')
        plt.title("Average Safety Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Safety Reward")
        plt.grid(True)
        
        # Plot average traffic flow per episode
        plt.subplot(4, 1, 4)
        if metrics_callback.episode_traffic_flow:
            avg_traffic = [np.mean(tf) if tf else 0 for tf in metrics_callback.episode_traffic_flow]
            plt.plot(avg_traffic, marker='o', color='purple')
            plt.title("Average Traffic Flow Per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Traffic Flow Reward")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("graphs", "reward_components.png"))
        plt.close()
        
        # 2) Plot the last episode in detail (timestep by timestep)
        if len(metrics_callback.episode_comfort) > 0:
            plt.figure(figsize=(15, 12))
            
            last_episode = -1
            try:
                timesteps = range(len(metrics_callback.episode_comfort[last_episode]))
                
                plt.subplot(4, 1, 1)
                plt.plot(timesteps, metrics_callback.episode_comfort[last_episode])
                plt.title(f"Final Episode Comfort (Episode {len(metrics_callback.episode_comfort)})")
                plt.xlabel("Timestep")
                plt.ylabel("Comfort Reward")
                plt.grid(True)
                
                plt.subplot(4, 1, 2)
                plt.plot(timesteps, metrics_callback.episode_efficiency[last_episode])
                plt.title("Final Episode Efficiency")
                plt.xlabel("Timestep")
                plt.ylabel("Efficiency Reward")
                plt.grid(True)
                
                plt.subplot(4, 1, 3)
                plt.plot(timesteps, metrics_callback.episode_safety[last_episode])
                plt.title("Final Episode Safety")
                plt.xlabel("Timestep")
                plt.ylabel("Safety Reward")
                plt.grid(True)
                
                plt.subplot(4, 1, 4)
                if metrics_callback.episode_traffic_flow and metrics_callback.episode_traffic_flow[last_episode]:
                    plt.plot(timesteps, metrics_callback.episode_traffic_flow[last_episode], color='purple')
                    plt.title("Final Episode Traffic Flow")
                    plt.xlabel("Timestep")
                    plt.ylabel("Traffic Flow Reward")
                    plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join("graphs", "final_episode_details.png"))
                plt.close()
            except Exception as e:
                print(f"Error plotting last episode details: {e}")
    
    # 3) Plot learning progress and lane changes
    plt.figure(figsize=(15, 10))
    
    # Plot Episode Rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics_callback.episode_rewards, marker='o')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot Episode Lengths
    plt.subplot(2, 2, 2)
    plt.plot(metrics_callback.episode_lengths, marker='o')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Plot Lane Changes per Episode
    plt.subplot(2, 2, 3)
    plt.plot(metrics_callback.episode_lane_changes, marker='o', color='green')
    plt.title('Lane Changes per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Lane Changes')
    plt.grid(True)
    
    # Plot Reward per Step (efficiency)
    plt.subplot(2, 2, 4)
    rewards_per_step = [r/l if l > 0 else 0 for r, l in zip(metrics_callback.episode_rewards, metrics_callback.episode_lengths)]
    plt.plot(rewards_per_step, marker='o', color='red')
    plt.title('Reward per Step')
    plt.xlabel('Episode')
    plt.ylabel('Reward/Step')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", "training_metrics.png"))
    plt.show()
    
    # Print final statistics
    if metrics_callback.episode_rewards:
        print("\nTraining Statistics:")
        print(f"Episodes completed: {len(metrics_callback.episode_rewards)}")
        print(f"Average Episode Reward: {np.mean(metrics_callback.episode_rewards):.2f}")
        print(f"Average Episode Length: {np.mean(metrics_callback.episode_lengths):.2f}")
        print(f"Average Lane Changes per Episode: {np.mean(metrics_callback.episode_lane_changes):.2f}")
        
        if len(metrics_callback.episode_rewards) >= 2:
            first_half = metrics_callback.episode_rewards[:len(metrics_callback.episode_rewards)//2]
            second_half = metrics_callback.episode_rewards[len(metrics_callback.episode_rewards)//2:]
            
            print(f"First half average reward: {np.mean(first_half):.2f}")
            print(f"Second half average reward: {np.mean(second_half):.2f}")
            if np.mean(second_half) > np.mean(first_half):
                print("The agent is showing signs of improvement!")

except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Always close environment to prevent SUMO from hanging
    try:
        env.close()
    except:
        pass