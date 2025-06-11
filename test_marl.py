import os
import matplotlib.pyplot as plt
import numpy as np
import traci
from stable_baselines3 import PPO
from custom_env import MultiAgentCarEnv
import time
import json

# Create output directories
os.makedirs("evaluation_logs", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# Constants
NUM_AGENTS = 3
NUM_EPISODES = 5

def load_models():
    """Load the trained models for each agent"""
    models = {}
    
    for i in range(NUM_AGENTS):
        agent_name = f'rlagent_{i}'
        model_path = f"models/ppo_agent{i}"
        
        try:
            model = PPO.load(model_path)
            print(f"Loaded model for {agent_name} from {model_path}")
            models[agent_name] = model
        except Exception as e:
            print(f"Error loading model for {agent_name}: {e}")
            return None
            
    return models

def evaluate_marl(models, n_episodes=NUM_EPISODES):
    """Run evaluation episodes with all trained agents"""
    if models is None:
        print("No models to evaluate.")
        return None
        
    # Create the environment
    env = MultiAgentCarEnv(num_agents=NUM_AGENTS)
    print("Starting evaluation environment...")
    
    # Track metrics for each agent
    metrics = {
        agent_name: {
            'episode_rewards': [],
            'episode_lengths': [],
            'comfort': [],
            'efficiency': [],
            'safety': [],
            'traffic_flow': [],
            'lane_changes': [],
            'collisions': [],
            'avg_speed': []
        } for agent_name in models.keys()
    }
    
    # Also track global metrics
    global_metrics = {
        'traffic_flow': [],
        'collision_rate': [],
        'avg_speed': [],
        'completion_time': []
    }
    
    print(f"\n===== Starting Evaluation =====")
    print(f"Will run {n_episodes} episodes with {NUM_AGENTS} agents.")
    
    for episode in range(n_episodes):
        print(f"\nStarting episode {episode + 1}/{n_episodes}")
        start_time = time.time()
        
        try:
            # Start with a new traffic scenario
            num_vehicles = np.random.randint(20, 35)
            success = env.start(gui=True, numVehicles=num_vehicles, vType='human')
            
            if not success:
                print(f"Error starting episode {episode + 1}. Skipping.")
                continue
                
            # Reset environment
            try:
                observations, info = env.reset(options={'numVehicles': num_vehicles})
                print(f"Environment reset successful. Active agents: {len(env.agents)}")
                if set(models.keys()) - set(env.agents):
                    print(f"Warning: Some agents not available after reset: {set(models.keys()) - set(env.agents)}")
            except Exception as e:
                print(f"Error during environment reset: {e}")
                continue
            
            # Initialize episode tracking
            episode_rewards = {agent: 0 for agent in env.possible_agents}
            episode_steps = 0
            episode_done = False
            all_infos = {agent: [] for agent in env.possible_agents}
            lane_changes = {agent: 0 for agent in env.possible_agents}
            prev_lanes = {agent: None for agent in env.possible_agents}
            
            # Display initial state visualization
            display_state(env, models, episode_rewards, 0)
            
            # Run the episode
            while not episode_done and episode_steps < env.max_steps:
                # Collect actions from all agents
                actions = {}
                
                for agent in env.agents:
                    if agent in observations and agent in models:
                        # Get action from policy
                        try:
                            action, _ = models[agent].predict(observations[agent], deterministic=True)
                            actions[agent] = action
                        except Exception as e:
                            print(f"Error getting action for {agent}: {e}")
                            # Use zero action as fallback
                            actions[agent] = np.zeros(env.action_spaces[agent].shape)
                
                # Step the environment
                try:
                    observations, rewards, terminations, truncations, infos = env.step(actions)
                except Exception as e:
                    print(f"Error during environment step: {e}")
                    episode_done = True
                    continue
                
                # Update episode tracking
                for agent in env.possible_agents:
                    if agent in rewards:
                        episode_rewards[agent] += rewards[agent]
                    
                    if agent in infos:
                        all_infos[agent].append(infos[agent])
                        
                        # Track lane changes
                        curr_lane = infos[agent].get("curr_sublane", None)
                        if prev_lanes[agent] is not None and curr_lane is not None and prev_lanes[agent] != curr_lane:
                            lane_changes[agent] += 1
                        prev_lanes[agent] = curr_lane
                
                # Check if episode is done (all agents terminated or reached max steps)
                if not env.agents:  # All agents have been terminated
                    episode_done = True
                else:
                    episode_done = all(terminations.values()) or all(truncations.values()) or episode_steps >= env.max_steps
                
                episode_steps += 1
                
                # Print progress and display state periodically
                if episode_steps % 100 == 0:
                    print(f"Step {episode_steps}, Rewards: {episode_rewards}")
                    display_state(env, models, episode_rewards, episode_steps)
            
            # Record episode duration
            episode_duration = time.time() - start_time
            
            # Record metrics for each agent
            for agent in env.possible_agents:
                if agent in models:
                    agent_infos = all_infos[agent]
                    
                    if not agent_infos:
                        print(f"Warning: No data collected for {agent} in episode {episode + 1}")
                        continue
                        
                    # Basic episode metrics
                    metrics[agent]['episode_rewards'].append(episode_rewards[agent])
                    metrics[agent]['episode_lengths'].append(episode_steps)
                    metrics[agent]['lane_changes'].append(lane_changes[agent])
                    
                    # Agent experienced a collision?
                    collision = any(info.get("collision", False) for info in agent_infos)
                    metrics[agent]['collisions'].append(1 if collision else 0)
                    
                    # Average component rewards (with safety checks for empty lists)
                    metrics[agent]['comfort'].append(np.mean([info.get("r_comf", 0) for info in agent_infos]) if agent_infos else 0)
                    metrics[agent]['efficiency'].append(np.mean([info.get("r_eff", 0) for info in agent_infos]) if agent_infos else 0)
                    metrics[agent]['safety'].append(np.mean([info.get("r_safe", 0) for info in agent_infos]) if agent_infos else 0)
                    metrics[agent]['traffic_flow'].append(np.mean([info.get("r_traffic", 0) for info in agent_infos]) if agent_infos else 0)
                    
                    # Average speed
                    metrics[agent]['avg_speed'].append(np.mean([info.get("speed", 0) for info in agent_infos]) if agent_infos else 0)
            
            # Calculate global metrics (with safety checks for empty values)
            collision_occurred = any(metrics[agent]['collisions'][-1] for agent in models.keys() if len(metrics[agent]['collisions']) > 0)
            global_metrics['collision_rate'].append(1 if collision_occurred else 0)
            global_metrics['completion_time'].append(episode_duration)
            
            # Calculate average speed across all agents
            agent_speeds = [metrics[agent]['avg_speed'][-1] for agent in models.keys() 
                            if len(metrics[agent]['avg_speed']) > 0]
            if agent_speeds:
                global_metrics['avg_speed'].append(np.mean(agent_speeds))
            else:
                global_metrics['avg_speed'].append(0)
            
            # Calculate overall traffic flow
            traffic_flows = [metrics[agent]['traffic_flow'][-1] for agent in models.keys() 
                             if len(metrics[agent]['traffic_flow']) > 0]
            if traffic_flows:
                global_metrics['traffic_flow'].append(np.mean(traffic_flows))
            else:
                global_metrics['traffic_flow'].append(0)
            
            # Print episode summary
            print(f"Episode {episode + 1} completed in {episode_duration:.1f} seconds")
            print(f"Steps: {episode_steps}, Collision: {collision_occurred}")
            for agent in models.keys():
                if len(metrics[agent]['episode_rewards']) > 0:
                    print(f"{agent}: Reward={episode_rewards.get(agent, 0):.2f}, Lane Changes={lane_changes.get(agent, 0)}")
                
            # Pause between episodes
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # Ensure we close SUMO at the end of each episode
            try:
                env.close()
                print("Environment closed successfully for this episode")
            except:
                print("Warning: Error when closing environment")
    
    try:
        # Make sure environment is closed
        env.close()
    except:
        pass
    
    # Save metrics to file
    with open(os.path.join("evaluation_logs", "evaluation_metrics.json"), 'w') as f:
        results = {
            "agent_metrics": metrics,
            "global_metrics": global_metrics
        }
        json.dump(results, f)
    
    # Plot evaluation results
    plot_evaluation_results(metrics, global_metrics)
    
    return metrics, global_metrics

def display_state(env, models, rewards, step):
    """Display current state information for debugging"""
    if not hasattr(env, 'agents') or not env.agents:
        print("No active agents to display state for")
        return
        
    print(f"\n--- State at step {step} ---")
    print(f"Active agents: {len(env.agents)} / {len(env.possible_agents)}")
    
    # Display reward information
    if rewards:
        print("Current rewards:")
        for agent, reward in rewards.items():
            if agent in env.agents:
                print(f"  {agent}: {reward:.2f}")
    
    # Display position and speed information if available
    if hasattr(env, 'pos') and hasattr(env, 'speed'):
        print("Agent positions and speeds:")
        for agent in env.agents:
            if agent in env.pos and agent in env.speed:
                pos = env.pos[agent]
                speed = env.speed[agent]
                curr_lane = env.curr_lane.get(agent, "unknown") if hasattr(env, 'curr_lane') else "unknown"
                print(f"  {agent}: pos=({pos[0]:.1f}, {pos[1]:.1f}), speed={speed:.1f} m/s, lane={curr_lane}")
    
    print("-------------------")

def plot_evaluation_results(metrics, global_metrics):
    """Plot evaluation results for all agents"""
    # Plot agent-specific metrics
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    for agent, agent_metrics in metrics.items():
        if agent_metrics['episode_rewards']:  # Only plot if we have data
            plt.plot(agent_metrics['episode_rewards'], label=agent, marker='o')
    plt.title('Evaluation Rewards by Agent')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot lane changes
    plt.subplot(2, 2, 2)
    for agent, agent_metrics in metrics.items():
        if agent_metrics['lane_changes']:  # Only plot if we have data
            plt.plot(agent_metrics['lane_changes'], label=agent, marker='o')
    plt.title('Lane Changes by Agent')
    plt.xlabel('Episode')
    plt.ylabel('Number of Lane Changes')
    plt.legend()
    plt.grid(True)
    
    # Plot average speed
    plt.subplot(2, 2, 3)
    for agent, agent_metrics in metrics.items():
        if agent_metrics['avg_speed']:  # Only plot if we have data
            plt.plot(agent_metrics['avg_speed'], label=agent, marker='o')
    plt.title('Average Speed by Agent')
    plt.xlabel('Episode')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid(True)
    
    # Plot collisions
    plt.subplot(2, 2, 4)
    for agent, agent_metrics in metrics.items():
        if agent_metrics['collisions']:  # Only plot if we have data
            plt.plot(agent_metrics['collisions'], label=agent, marker='o')
    plt.title('Collisions by Agent')
    plt.xlabel('Episode')
    plt.ylabel('Collision Occurred (1=Yes, 0=No)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", "evaluation_agent_metrics.png"))
    plt.close()
    
    # Plot global metrics
    plt.figure(figsize=(10, 8))
    
    # Only plot if we have data
    if global_metrics['collision_rate']:
        # Plot collision rate
        plt.subplot(2, 2, 1)
        plt.plot(global_metrics['collision_rate'], marker='o', color='red')
        plt.title('Overall Collision Rate')
        plt.xlabel('Episode')
        plt.ylabel('Collision Occurred (1=Yes, 0=No)')
        plt.grid(True)
    
    if global_metrics['completion_time']:
        # Plot completion time
        plt.subplot(2, 2, 2)
        plt.plot(global_metrics['completion_time'], marker='o', color='blue')
        plt.title('Episode Completion Time')
        plt.xlabel('Episode')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
    
    if global_metrics['avg_speed']:
        # Plot average speed
        plt.subplot(2, 2, 3)
        plt.plot(global_metrics['avg_speed'], marker='o', color='green')
        plt.title('Average Speed (All Agents)')
        plt.xlabel('Episode')
        plt.ylabel('Speed (m/s)')
        plt.grid(True)
    
    if global_metrics['traffic_flow']:
        # Plot traffic flow
        plt.subplot(2, 2, 4)
        plt.plot(global_metrics['traffic_flow'], marker='o', color='purple')
        plt.title('Overall Traffic Flow')
        plt.xlabel('Episode')
        plt.ylabel('Traffic Flow Score')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", "evaluation_global_metrics.png"))
    plt.close()
    
    # Print summary statistics
    print("\n===== Evaluation Summary =====")
    
    # Global metrics - only calculate if we have data
    if global_metrics['collision_rate']:
        collision_rate = np.mean(global_metrics['collision_rate']) * 100
        print(f"Overall collision rate: {collision_rate:.1f}%")
    else:
        print("Overall collision rate: N/A (no data)")
    
    if global_metrics['completion_time']:
        avg_completion_time = np.mean(global_metrics['completion_time'])
        print(f"Average episode completion time: {avg_completion_time:.1f} seconds")
    else:
        print("Average episode completion time: N/A (no data)")
    
    if global_metrics['avg_speed']:
        avg_global_speed = np.mean(global_metrics['avg_speed'])
        print(f"Average global speed: {avg_global_speed:.1f} m/s")
    else:
        print("Average global speed: N/A (no data)")
    
    # Per-agent metrics
    print("\nPer-agent metrics:")
    for agent, agent_metrics in metrics.items():
        print(f"\n{agent}:")
        
        if agent_metrics['episode_rewards']:
            avg_reward = np.mean(agent_metrics['episode_rewards'])
            print(f"  Average reward: {avg_reward:.2f}")
        else:
            print("  Average reward: N/A (no data)")
        
        if agent_metrics['lane_changes']:
            avg_lane_changes = np.mean(agent_metrics['lane_changes'])
            print(f"  Average lane changes: {avg_lane_changes:.1f}")
        else:
            print("  Average lane changes: N/A (no data)")
        
        if agent_metrics['avg_speed']:
            avg_speed = np.mean(agent_metrics['avg_speed'])
            print(f"  Average speed: {avg_speed:.1f} m/s")
        else:
            print("  Average speed: N/A (no data)")
        
        if agent_metrics['collisions']:
            collision_rate = np.mean(agent_metrics['collisions']) * 100
            print(f"  Collision rate: {collision_rate:.1f}%")
        else:
            print("  Collision rate: N/A (no data)")

if __name__ == "__main__":
    print("Loading trained models...")
    models = load_models()
    
    if models:
        print(f"Running evaluation with {len(models)} agents for {NUM_EPISODES} episodes...")
        evaluate_marl(models, NUM_EPISODES)
    else:
        print("No models to evaluate. Train agents first using train_marl.py.") 