import os
import matplotlib.pyplot as plt
import numpy as np
import traci
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces
import json
from custom_env import MultiAgentCarEnv
import time
from typing import Dict, List, Tuple

# Create folders for outputs
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# Constants
MODEL_DIR = "models"
LOG_DIR = "logs"
TOTAL_TIMESTEPS = 184320
CHECKPOINT_FREQ = 10000
NUM_AGENTS = 3  # Number of RL agents to use

# Multi-agent wrapper for parallel training
class MultiPPOTrainer:
    def __init__(self, env, num_agents=NUM_AGENTS, metrics_callback=None):
        self.env = env
        self.num_agents = num_agents
        self.metrics_callback = metrics_callback
        
        # Create agent IDs
        self.agent_ids = [f'rlagent_{i}' for i in range(num_agents)]
        
        # Initialize PPO models for each agent
        self.models = {}
        for agent_id in self.agent_ids:
            # Create a proper wrapper function that captures agent_id correctly
            def create_env_wrapper(agent_id=agent_id):  # Default argument fixes closure issue
                return AgentObservationWrapper(env, agent_id)
                
            self.models[agent_id] = PPO(
                "MlpPolicy",
                DummyVecEnv([create_env_wrapper]),  # Use the function that correctly captures agent_id
                verbose=0,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256])
                )
            )
            print(f"Initialized PPO model for {agent_id}")
        
        # Initialize training stats
        self.steps_per_agent = {agent_id: 0 for agent_id in self.agent_ids}
        self.episodes_per_agent = {agent_id: 0 for agent_id in self.agent_ids}
    
    def collect_experience(self, timesteps):
        """
        Run environment for specified timesteps, collecting experience for all agents.
        Returns the total reward and episode information for each agent.
        """
        # Set up metrics tracking
        episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        episode_lengths = {agent_id: 0 for agent_id in self.agent_ids}
        episode_infos = {agent_id: [] for agent_id in self.agent_ids}
        
        # Initialize episode states
        done = False
        timestep = 0
        episode_terminated = False
        
        # Storage for training data per agent
        experiences = {agent_id: [] for agent_id in self.agent_ids}
        
        # Use the current active agents in the environment
        active_agents = set(self.env.agents)
        
        # Get current observations from environment
        observations = {}
        for agent_id in active_agents:
            try:
                # Get observation for this agent
                obs = self.env.get_state(agent_name=agent_id)
                if obs is not None:
                    observations[agent_id] = obs
            except Exception as e:
                print(f"Warning: Could not get state for {agent_id}: {e}")
                continue
        
        while timestep < timesteps and not done and not episode_terminated:
            actions = {}
            
            # Get actions from each agent's policy
            for agent_id in active_agents:
                if agent_id in observations:
                    # Get observation for this agent
                    obs = observations[agent_id]
                    
                    # Predict action using the agent's policy
                    action, _ = self.models[agent_id].predict(obs, deterministic=False)
                    actions[agent_id] = action
            
            # Step the environment with all actions
            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Check if episode is naturally done
            if all(terminations.values()) or all(truncations.values()):
                episode_terminated = True
                print("Episode naturally terminated during experience collection.")
            
            # Update active agents list based on environment's current state
            active_agents = set(self.env.agents)
            
            # Store experiences for each agent
            for agent_id in self.agent_ids:
                # Only process agents that were in both observations and next_observations
                if agent_id in observations and agent_id in next_observations:
                    # Store transitions for this agent
                    experiences[agent_id].append((
                        observations[agent_id],
                        actions.get(agent_id, np.zeros(self.env.action_spaces[agent_id].shape)),
                        rewards.get(agent_id, 0.0),
                        next_observations[agent_id],
                        terminations.get(agent_id, False) or truncations.get(agent_id, False)
                    ))
                    
                    # Update episode metrics
                    episode_rewards[agent_id] += rewards.get(agent_id, 0.0)
                    episode_lengths[agent_id] += 1
                    episode_infos[agent_id].append(infos.get(agent_id, {}))
            
            # Update observations for next step
            observations = next_observations
            
            # Increment timestep
            timestep += 1
            
            # Periodically log progress
            if timestep % 100 == 0:
                rewards_str = ", ".join([f"{a}: {r:.2f}" for a, r in episode_rewards.items()])
                print(f"Step {timestep}/{timesteps} - Rewards: {rewards_str}")
                
                # Check if environment might have been reset
                if hasattr(self.env, 'curr_step') and self.env.curr_step < timestep:
                    print("Warning: Environment appears to have reset during experience collection!")
                    # If SUMO restarted unexpectedly, we'll return what we have so far
                    break
        
        if timestep < timesteps:
            print(f"Collected {timestep}/{timesteps} steps before episode completion or interruption.")
        
        return episode_rewards, episode_lengths, episode_infos, experiences
    
    def update_policies(self, experiences):
        """Update all agent policies with their collected experiences"""
        for agent_id, agent_experiences in experiences.items():
            if not agent_experiences:
                continue
                
            # Prepare data in the format required by PPO
            # This is a simplified version - in practice, you'd need proper rollout buffer integration
            observations, actions, rewards, next_obs, dones = zip(*agent_experiences)
            
            # Manual mini-batch learning (simplified)
            # In practice, use the PPO's learn() method with proper buffers
            self.models[agent_id].learn(total_timesteps=1, reset_num_timesteps=False)
            
            # Update steps
            self.steps_per_agent[agent_id] += len(agent_experiences)
    
    def train(self, total_timesteps, eval_freq=10000):
        """
        Train all agents simultaneously for the specified number of timesteps.
        """
        timesteps_per_iteration = 2048 
        iterations = total_timesteps // timesteps_per_iteration
        
        # Create system metrics callback if metrics_callback is provided
        system_callback = None
        if self.metrics_callback is not None:
            from stable_baselines3.common.callbacks import CallbackList
            system_callback = SystemMetricsCallback(self.metrics_callback)
        
        print(f"\n===== Training for {iterations} iterations =====")
        
        # Initialize environment once at the start with force_restart=True to ensure clean startup
        observations, _ = self.env.reset(options={'force_restart': True})
        
        for iteration in range(iterations):
            print(f"\n===== Iteration {iteration+1}/{iterations} =====")
            
            # Fresh start for each iteration - always restart SUMO after the first iteration
            if iteration > 0:
                print("Restarting SUMO for new iteration")
                # Explicitly close the environment to ensure SUMO is completely shut down
                try:
                    self.env.close()
                    # Give system time to fully clean up resources
                    time.sleep(1.0)
                except Exception as e:
                    print(f"Warning during environment close: {e}")
                
                # Force a complete restart of SUMO by setting force_restart=True
                observations, _ = self.env.reset(options={'force_restart': True, 'gui': True})
            
            # Collect experiences for all agents
            rewards, lengths, infos, experiences = self.collect_experience(timesteps_per_iteration)
            
            # Update all agents' policies with collected experiences
            self.update_policies(experiences)
            
            # Report progress
            rewards_str = ", ".join([f"{a}: {r:.2f}" for a, r in rewards.items()])
            print(f"Iteration {iteration+1} complete - Episode rewards: {rewards_str}")
            
            # Update episode counters
            for agent_id in self.agent_ids:
                if rewards[agent_id] != 0:  # If agent participated in this episode
                    self.episodes_per_agent[agent_id] += 1
            
            # Save models periodically
            if (iteration + 1) % 5 == 0:
                self.save_models()
            
            # Update metrics callback if provided
            if self.metrics_callback is not None:
                # Update metrics for each agent
                for agent_id in self.agent_ids:
                    agent_idx = int(agent_id.split('_')[1])
                    
                    # Update metrics
                    self.metrics_callback.current_episode_reward[agent_idx] = rewards[agent_id]
                    self.metrics_callback.current_episode_length[agent_idx] = lengths[agent_id]
                    
                    # Process infos for this agent
                    agent_infos = infos[agent_id]
                    all_infos = {}
                    
                    # Construct combined info dictionary for system metrics
                    for info in agent_infos:
                        all_infos[agent_id] = info
                    
                    # Process reward components
                    for info in agent_infos:
                        if 'r_comf' in info:
                            self.metrics_callback.current_episode_comfort[agent_idx].append(info['r_comf'])
                        if 'r_eff' in info:
                            self.metrics_callback.current_episode_efficiency[agent_idx].append(info['r_eff'])
                        if 'r_safe' in info:
                            self.metrics_callback.current_episode_safety[agent_idx].append(info['r_safe'])
                        if 'r_traffic' in info:
                            self.metrics_callback.current_episode_traffic_flow[agent_idx].append(info['r_traffic'])
                    
                    # Simulate end of episode for metrics
                    self.metrics_callback.episode_rewards[agent_idx].append(rewards[agent_id])
                    self.metrics_callback.episode_lengths[agent_idx].append(lengths[agent_id])
                    
                    # Reset for next iteration
                    self.metrics_callback.current_episode_reward[agent_idx] = 0.0
                    self.metrics_callback.current_episode_length[agent_idx] = 0
                    
                    # Update training time
                    self.metrics_callback.training_times[agent_idx] += lengths[agent_id] * 0.01  # approximate time
                
                # Update system metrics
                if system_callback is not None and all_infos:
                    system_callback.metrics_callback.calculate_system_performance(all_infos)
                
                # Create plots every 5 episodes
                if sum(self.episodes_per_agent.values()) % 5 == 0:
                    self.metrics_callback.create_plots()
        
        # Final save
        self.save_models()
        
        return self.models
    
    def save_models(self):
        """Save all agent models"""
        for agent_id, model in self.models.items():
            agent_idx = int(agent_id.split('_')[1])
            path = os.path.join(MODEL_DIR, f"ppo_agent{agent_idx}")
            model.save(path)
            print(f"Saved model for {agent_id} to {path}")

# Observation wrapper for a single agent
class AgentObservationWrapper(gym.Env):
    """
    Wraps the multi-agent environment to provide a single-agent interface for a specific agent.
    Used for PPO compatibility with the multi-agent environment.
    """
    def __init__(self, env, agent_id):
        self.env = env
        self.agent_id = agent_id
        
        # Define the observation and action spaces for this agent
        self.observation_space = env.observation_spaces[agent_id]
        self.action_space = env.action_spaces[agent_id]
        
        # Required for Gymnasium Env
        self.metadata = {"render_modes": ["human"]}
        
    def reset(self, **kwargs):
        obs_dict, info_dict = self.env.reset(**kwargs)
        obs = obs_dict.get(self.agent_id, np.zeros(self.observation_space.shape))
        info = info_dict.get(self.agent_id, {})
        return obs, info
    
    def step(self, action):
        # Create action dict for the MARL environment
        actions = {self.agent_id: action}
        
        # Fill in actions for other agents with zeros
        for other_agent in self.env.agents:
            if other_agent != self.agent_id:
                actions[other_agent] = np.zeros(self.env.action_spaces[other_agent].shape)
        
        # Step the environment
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(actions)
        
        # Extract results for this agent
        obs = obs_dict.get(self.agent_id, np.zeros(self.observation_space.shape))
        rew = rew_dict.get(self.agent_id, 0.0)
        term = term_dict.get(self.agent_id, False)
        trunc = trunc_dict.get(self.agent_id, False)
        info = info_dict.get(self.agent_id, {})
        
        return obs, rew, term, trunc, info
        
    def render(self):
        # Pass through to env's render method if it exists
        if hasattr(self.env, 'render'):
            return self.env.render()
    
    def close(self):
        # Pass through to env's close method if it exists
        if hasattr(self.env, 'close'):
            return self.env.close()

# Custom callback for collecting metrics - redesigned for simultaneous multi-agent training
class MetricsCallback(BaseCallback):
    def __init__(self, num_agents=NUM_AGENTS, verbose=0):
        super().__init__(verbose)
        self.num_agents = num_agents
        
        # Episode-level metrics for each agent - using native Python types
        self.episode_rewards = {i: [] for i in range(num_agents)}
        self.episode_lengths = {i: [] for i in range(num_agents)}
        self.current_episode_reward = {i: 0.0 for i in range(num_agents)}
        self.current_episode_length = {i: 0 for i in range(num_agents)}
        
        # Components of reward for each agent
        self.episode_comfort = {i: [] for i in range(num_agents)}
        self.episode_efficiency = {i: [] for i in range(num_agents)}
        self.episode_safety = {i: [] for i in range(num_agents)}
        self.episode_lane_changes = {i: [] for i in range(num_agents)}
        self.episode_traffic_flow = {i: [] for i in range(num_agents)}
        self.current_episode_comfort = {i: [] for i in range(num_agents)}
        self.current_episode_efficiency = {i: [] for i in range(num_agents)}
        self.current_episode_safety = {i: [] for i in range(num_agents)}
        self.current_episode_traffic_flow = {i: [] for i in range(num_agents)}
        
        # Agent-specific indices for plotting
        self.safety_indices = {i: [] for i in range(num_agents)}
        self.efficiency_indices = {i: [] for i in range(num_agents)}
        
        # For tracking progress
        self.n_episodes = {i: 0 for i in range(num_agents)}
        self.lane_change_count = {i: 0 for i in range(num_agents)}
        self.prev_lane = {i: None for i in range(num_agents)}
        
        # Global metrics - combine data across all agents
        self.global_rewards = []
        self.global_lane_changes = []
        self.global_collisions = []
        self.global_step_count = 0
        self.collision_count = {i: 0 for i in range(num_agents)}
        self.episode_collisions = {i: [] for i in range(num_agents)}
        
        # System performance metrics
        self.system_performance_index = []  # Combined performance across all agents
        self.safety_index = []  # System-wide safety performance
        self.efficiency_index = []  # System-wide traffic efficiency
        self.coordination_index = []  # Measure of agent coordination
        
        # Store current system state
        self.current_safety_index = 0.0
        self.current_efficiency_index = 0.0
        self.current_coordination_index = 0.0
        self.current_system_performance = 0.0
        self.system_metrics_counter = 0
        
        # Last observation time for each agent to track reporting frequency
        self.last_report_time = {i: 0 for i in range(num_agents)}
        
        # Record training start time
        self.start_time = time.time()
        self.training_times = {i: 0.0 for i in range(num_agents)}
        
        # Track last agent for which we created plots (only used for backward compatibility)
        self.last_plotting_agent = 0
        
    def set_current_agent(self, agent_idx):
        """
        Compatibility method for transitioning to simultaneous training.
        In true simultaneous training, we don't need this, but keeping it for backward compatibility.
        """
        # Just store this agent as the last one we operated on (for plotting)
        self.last_plotting_agent = agent_idx
        
        # Display training summary for all agents
        self.display_all_agent_stats()
        
        # For tracking time
        self.start_time = time.time()
        
        print(f"\n--- Agent {agent_idx} is now the focus for plotting ---")
    
    def display_all_agent_stats(self):
        """Display a summary of training statistics for all agents"""
        print("\n===== MULTI-AGENT TRAINING SUMMARY =====")
        
        # Table header
        print(f"{'Agent':^8}|{'Episodes':^10}|{'Avg Reward':^12}|{'Avg Length':^12}|{'Lane Changes':^14}|{'Collisions':^12}|{'Training Time':^15}")
        print("-" * 85)
        
        # Show data for each agent
        for i in range(self.num_agents):
            avg_reward = sum(self.episode_rewards[i]) / len(self.episode_rewards[i]) if self.episode_rewards[i] else 0.0
            avg_length = sum(self.episode_lengths[i]) / len(self.episode_lengths[i]) if self.episode_lengths[i] else 0.0
            total_lane_changes = sum(self.episode_lane_changes[i])
            total_collisions = sum(self.episode_collisions[i])
            training_time = self.training_times[i]
            
            time_str = f"{training_time:.1f}s" if training_time < 60 else f"{training_time/60:.1f}m"
            
            print(f"{i:^8}|{self.n_episodes[i]:^10}|{avg_reward:^12.2f}|{avg_length:^12.1f}|{total_lane_changes:^14}|{total_collisions:^12}|{time_str:^15}")
        
        print("=" * 85)
        
        # If we have system performance data, display it
        if len(self.system_performance_index) > 0:
            print("\n----- SYSTEM PERFORMANCE METRICS -----")
            avg_performance = sum(self.system_performance_index) / len(self.system_performance_index)
            avg_safety = sum(self.safety_index) / len(self.safety_index)
            avg_efficiency = sum(self.efficiency_index) / len(self.efficiency_index) 
            avg_coordination = sum(self.coordination_index) / len(self.coordination_index)
            
            print(f"Overall System Performance Index: {avg_performance:.2f}")
            print(f"Safety Index: {avg_safety:.2f}")
            print(f"Efficiency Index: {avg_efficiency:.2f}")
            print(f"Coordination Index: {avg_coordination:.2f}")
            print("-" * 40)
    
    def _on_step(self):
        # Get the all_agent_infos that contains data for all agents
        info = self.locals["infos"][0]  # The first env info
        all_agent_infos = info.get("all_agent_infos", {})
        
        # If we have no multi-agent data, return early
        if not all_agent_infos:
            return True
            
        # Update global step counter
        self.global_step_count += 1
        
        # Process data for all agents simultaneously
        for agent_name, agent_info in all_agent_infos.items():
            # Extract agent index from name (format: 'rlagent_X')
            try:
                agent_idx = int(agent_name.split('_')[1])
                if agent_idx >= self.num_agents:
                    continue  # Skip if agent index is out of range
            except (ValueError, IndexError):
                continue  # Skip if agent name doesn't match expected format
                
            # Extract reward - need to find it from the rewards dict in locals
            rewards_dict = self.locals.get("rewards", [{}])[0]
            if isinstance(rewards_dict, dict) and agent_name in rewards_dict:
                reward = float(rewards_dict[agent_name])
            else:
                reward = 0.0  # Default if reward not found
                
            # Update agent metrics
            self.current_episode_reward[agent_idx] += reward
            self.current_episode_length[agent_idx] += 1
            
            # Extract reward components from info dictionary
            r_safe = float(agent_info.get("r_safe", 0.0))
            r_eff = float(agent_info.get("r_eff", 0.0))
            
            # Convert reward components to indices (0 to 1 range)
            safety_index = (r_safe + 1) / 2  # Convert from [-1,1] to [0,1]
            efficiency_index = (r_eff + 1) / 2
            
            # Store indices for plotting
            if not hasattr(self, 'safety_indices'):
                self.safety_indices = {i: [] for i in range(self.num_agents)}
            if not hasattr(self, 'efficiency_indices'):
                self.efficiency_indices = {i: [] for i in range(self.num_agents)}
            
            # Store the current indices
            self.current_episode_safety[agent_idx].append(safety_index)
            self.current_episode_efficiency[agent_idx].append(efficiency_index)
            
            # Check for collision
            collision_occurred = agent_info.get("collision", False)
            if collision_occurred:
                self.collision_count[agent_idx] += 1
                
            # Track lane changes
            curr_lane = agent_info.get("curr_lane", None)
            if self.prev_lane[agent_idx] is not None and curr_lane is not None and self.prev_lane[agent_idx] != curr_lane:
                self.lane_change_count[agent_idx] += 1
            self.prev_lane[agent_idx] = curr_lane
            
            # Determine which agents to report in this step (based on time)
            current_time = time.time()
            
            # Report agent metrics periodically (every 100 steps per agent, or if 300 seconds elapsed)
            should_report = (
                self.current_episode_length[agent_idx] % 100 == 0 or 
                current_time - self.last_report_time.get(agent_idx, 0) > 300
            )
            
            if should_report:
                elapsed_time = current_time - self.start_time
                self.last_report_time[agent_idx] = current_time
                
                print(f"\n[{elapsed_time:.1f}s] Agent {agent_idx} - Step {self.current_episode_length[agent_idx]}")
                print(f"  Safety Index: {safety_index:.2f}, Efficiency Index: {efficiency_index:.2f}")
                print(f"  Total reward: {self.current_episode_reward[agent_idx]:.2f}")
                print(f"  Status: Lane changes={self.lane_change_count[agent_idx]}, Collisions={self.collision_count[agent_idx]}")
        
        # Check for episode end
        # We need to check if ANY agent has terminated, not just the current one
        dones = self.locals.get("dones", [False])[0]
        truncateds = self.locals.get("truncated", False)
        
        # For multi-agent environments, dones could be a dict
        if isinstance(dones, dict):
            done = any(dones.values())
        else:
            done = dones
            
        # Similar for truncated
        if isinstance(truncateds, dict):
            truncated = any(truncateds.values())
        else:
            truncated = truncateds
            
        episode_ended = done or truncated
        
        if episode_ended:
            # For all agents, finalize the episode
            for agent_idx in range(self.num_agents):
                if self.current_episode_length[agent_idx] > 0:  # Only process agents that participated
                    self.n_episodes[agent_idx] += 1
                    
                    # Store metrics for this agent
                    self.episode_rewards[agent_idx].append(self.current_episode_reward[agent_idx])
                    self.episode_lengths[agent_idx].append(self.current_episode_length[agent_idx])
                    
                    # Store component histories
                    self.episode_comfort[agent_idx].append(self.current_episode_comfort[agent_idx].copy())
                    self.episode_efficiency[agent_idx].append(self.current_episode_efficiency[agent_idx].copy())
                    self.episode_safety[agent_idx].append(self.current_episode_safety[agent_idx].copy())
                    self.episode_traffic_flow[agent_idx].append(self.current_episode_traffic_flow[agent_idx].copy())
                    self.episode_lane_changes[agent_idx].append(self.lane_change_count[agent_idx])
                    self.episode_collisions[agent_idx].append(self.collision_count[agent_idx])
                    
                    # Add to global metrics
                    self.global_rewards.append(self.current_episode_reward[agent_idx])
                    self.global_lane_changes.append(self.lane_change_count[agent_idx])
                    self.global_collisions.append(self.collision_count[agent_idx])
                    
                    # Print episode summary for each agent
                    print(f"\n==== Agent {agent_idx} - Episode {self.n_episodes[agent_idx]} Summary ====")
                    print(f"  Episode reward: {self.current_episode_reward[agent_idx]:.2f}")
                    print(f"  Episode length: {self.current_episode_length[agent_idx]} steps")
                    print(f"  Lane changes: {self.lane_change_count[agent_idx]}")
                    print(f"  Collisions: {self.collision_count[agent_idx]}")
                    
                    # Add overall performance metrics
                    # Calculate average of last 10 episodes or all episodes if fewer than 10
                    recent_rewards = self.episode_rewards[agent_idx][-10:] if len(self.episode_rewards[agent_idx]) >= 10 else self.episode_rewards[agent_idx]
                    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
                    print(f"  Average reward (last 10 episodes): {avg_reward:.2f}")
                    
                    # Reset for next episode
                    self.current_episode_reward[agent_idx] = 0.0
                    self.current_episode_length[agent_idx] = 0
                    self.current_episode_comfort[agent_idx] = []
                    self.current_episode_efficiency[agent_idx] = []
                    self.current_episode_safety[agent_idx] = []
                    self.current_episode_traffic_flow[agent_idx] = []
                    self.lane_change_count[agent_idx] = 0
                    self.collision_count[agent_idx] = 0
                    self.prev_lane[agent_idx] = None
            
            # Calculate system performance metrics
            system_metrics = self.calculate_system_performance(all_agent_infos)
            
            # Print system performance
            if system_metrics:
                print("\n----- SYSTEM PERFORMANCE -----")
                print(f"  Combined Performance Index: {system_metrics['system_index']:.2f}")
                print(f"  Safety Index: {system_metrics['safety']:.2f}")
                print(f"  Efficiency Index: {system_metrics['efficiency']:.2f}")
                print(f"  Coordination Index: {system_metrics['coordination']:.2f}")
                print("-----------------------------")
            
            # Update training times for all agents
            current_time = time.time()
            # Divide time equally among agents in simultaneous training
            time_per_agent = (current_time - self.start_time) / self.num_agents
            for agent_idx in range(self.num_agents):
                self.training_times[agent_idx] += time_per_agent
            
            # Create plots every 5 episodes
            if sum(self.n_episodes.values()) % 5 == 0:
                self.create_plots()
                
        return True

    def create_plots(self):
        """Create and save plots for all agents and global metrics"""
        # If last_plotting_agent isn't valid, default to agent 0
        if not hasattr(self, 'last_plotting_agent') or self.last_plotting_agent >= self.num_agents:
            self.last_plotting_agent = 0
            
        # Create plots for each agent individually
        for agent_idx in range(self.num_agents):
            if not self.episode_rewards[agent_idx]:
                continue  # Skip agents with no data
                
            # Individual agent plots
            plt.figure(figsize=(15, 10))
            
            # Plot rewards
            plt.subplot(2, 2, 1)
            plt.plot(self.episode_rewards[agent_idx], marker='o')
            plt.title(f'Agent {agent_idx} - Training Rewards (Ep {len(self.episode_rewards[agent_idx])})')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            
            # Plot episode lengths
            plt.subplot(2, 2, 2)
            plt.plot(self.episode_lengths[agent_idx], marker='o')
            plt.title(f'Agent {agent_idx} - Episode Lengths')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True)
            
            # Plot safety index 
            plt.subplot(2, 2, 3)
            if self.episode_safety[agent_idx]:
                # Calculate averages using native Python
                avg_safety = []
                
                for comps in self.episode_safety[agent_idx]:
                    if comps:  # Check if we have data
                        avg = sum(comps) / len(comps)
                        avg_safety.append(avg)  # Already in 0-1 range from _on_step
                
                if avg_safety:  # Only plot if we have data
                    plt.plot(avg_safety, marker='o', color='green')
                    plt.title(f'Agent {agent_idx} - Safety Index')
                    plt.xlabel('Episode')
                    plt.ylabel('Safety Index (0-1)')
                    plt.ylim(0, 1)
                    plt.grid(True)
            
            # Plot efficiency index
            plt.subplot(2, 2, 4)
            if self.episode_efficiency[agent_idx]:
                # Calculate averages using native Python
                avg_efficiency = []
                
                for comps in self.episode_efficiency[agent_idx]:
                    if comps:  # Check if we have data
                        avg = sum(comps) / len(comps)
                        avg_efficiency.append(avg)  # Already in 0-1 range from _on_step
                
                if avg_efficiency:  # Only plot if we have data
                    plt.plot(avg_efficiency, marker='o', color='blue')
                    plt.title(f'Agent {agent_idx} - Efficiency Index')
                    plt.xlabel('Episode')
                    plt.ylabel('Efficiency Index (0-1)')
                    plt.ylim(0, 1)
                    plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join("graphs", f"agent{agent_idx}_training_ep{len(self.episode_rewards[agent_idx])}.png"))
            plt.close()
        
        # Plot system performance metrics if available
        if len(self.system_performance_index) > 0:
            plt.figure(figsize=(15, 10))
            
            # Plot system performance index
            plt.subplot(2, 2, 1)
            plt.plot(self.system_performance_index, marker='o', color='purple', linewidth=2)
            plt.title('System Performance Index', fontsize=12, fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel('Index Value')
            plt.grid(True)
            
            # Plot safety index
            plt.subplot(2, 2, 2)
            plt.plot(self.safety_index, marker='o', color='green', linewidth=2)
            plt.title('Safety Index', fontsize=12, fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel('Index Value (0-1)')
            plt.ylim(0, 1)
            plt.grid(True)
            
            # Plot efficiency index
            plt.subplot(2, 2, 3)
            plt.plot(self.efficiency_index, marker='o', color='blue', linewidth=2)
            plt.title('Efficiency Index', fontsize=12, fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel('Index Value (0-1)')
            plt.ylim(0, 1)
            plt.grid(True)
            
            # Plot coordination index
            plt.subplot(2, 2, 4)
            plt.plot(self.coordination_index, marker='o', color='orange', linewidth=2)
            plt.title('Coordination Index', fontsize=12, fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel('Index Value (0-1)')
            plt.ylim(0, 1)
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join("graphs", f"system_performance_metrics_ep{len(self.system_performance_index)}.png"))
            plt.close()
        
        # Create global comparative plots for all agents
        self.plot_global_metrics()
    
    def plot_global_metrics(self):
        """Plot global metrics comparing all agents"""
        # Get current timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        plt.figure(figsize=(15, 10))
        
        # Plot rewards for all agents
        plt.subplot(2, 2, 1)
        for i in range(self.num_agents):
            if self.episode_rewards[i]:  # Only plot if we have data
                plt.plot(self.episode_rewards[i], label=f'Agent {i}')
        plt.title('Training Rewards by Agent')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        
        # Plot episode lengths for all agents
        plt.subplot(2, 2, 2)
        for i in range(self.num_agents):
            if self.episode_lengths[i]:  # Only plot if we have data
                plt.plot(self.episode_lengths[i], label=f'Agent {i}')
        plt.title('Episode Lengths by Agent')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()
        plt.grid(True)
        
        # Plot safety index - separate graph
        plt.subplot(2, 2, 3)
        for i in range(self.num_agents):
            if hasattr(self, 'safety_indices') and self.safety_indices[i]:
                plt.plot(self.safety_indices[i], label=f'Agent {i}')
        plt.title('Safety Index by Agent')
        plt.xlabel('Step')
        plt.ylabel('Safety Index (0-1)')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        # Plot efficiency index - separate graph
        plt.subplot(2, 2, 4)
        for i in range(self.num_agents):
            if hasattr(self, 'efficiency_indices') and self.efficiency_indices[i]:
                plt.plot(self.efficiency_indices[i], label=f'Agent {i}')
        plt.title('Efficiency Index by Agent')
        plt.xlabel('Step')
        plt.ylabel('Efficiency Index (0-1)')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("graphs", f"global_metrics_comparison_{timestamp}.png"))
        plt.close()

    def track_non_learning_agent(self, agent_idx, reward, info, step_count):
        """Track metrics for agents that are not currently being trained"""
        # Update cumulative reward
        self.current_episode_reward[agent_idx] += float(reward)
        self.current_episode_length[agent_idx] += 1
        
        # Check for collision
        collision = info.get("collision", False)
        if collision:
            self.collision_count[agent_idx] += 1
            
        # Track lane changes - current lane vs previous
        curr_lane = info.get("curr_lane", None)
        if self.prev_lane[agent_idx] is not None and curr_lane is not None and self.prev_lane[agent_idx] != curr_lane:
            self.lane_change_count[agent_idx] += 1
            if self.lane_change_count[agent_idx] % 20 == 0:  # Less frequent updates for non-learning agents
                print(f"[INFO] Agent {agent_idx} made {self.lane_change_count[agent_idx]} lane changes (non-learning)")
                
        self.prev_lane[agent_idx] = curr_lane
        
        # Track reward components
        comf = float(info.get("r_comf", 0.0))
        eff = float(info.get("r_eff", 0.0))
        safe = float(info.get("r_safe", 0.0))
        traffic = float(info.get("r_traffic", 0.0))
        
        # Log components
        self.current_episode_comfort[agent_idx].append(comf)
        self.current_episode_efficiency[agent_idx].append(eff)
        self.current_episode_safety[agent_idx].append(safe)
        self.current_episode_traffic_flow[agent_idx].append(traffic)
        
        # Handle episode end (simulated) - every 10,000 steps to match typical episode length
        if step_count > 0 and step_count % 10000 == 0:
            # Record episode metrics
            self.n_episodes[agent_idx] += 1
            self.episode_rewards[agent_idx].append(self.current_episode_reward[agent_idx])
            self.episode_lengths[agent_idx].append(self.current_episode_length[agent_idx])
            self.episode_comfort[agent_idx].append(self.current_episode_comfort[agent_idx].copy())
            self.episode_efficiency[agent_idx].append(self.current_episode_efficiency[agent_idx].copy())
            self.episode_safety[agent_idx].append(self.current_episode_safety[agent_idx].copy())
            self.episode_traffic_flow[agent_idx].append(self.current_episode_traffic_flow[agent_idx].copy())
            self.episode_lane_changes[agent_idx].append(self.lane_change_count[agent_idx])
            self.episode_collisions[agent_idx].append(self.collision_count[agent_idx])
            
            # Print summary for the non-learning agent
            print(f"\n==== Agent {agent_idx} (Non-Learning) - Episode Summary ====")
            print(f"  Episode reward: {self.current_episode_reward[agent_idx]:.2f}")
            print(f"  Episode length: {self.current_episode_length[agent_idx]} steps")
            print(f"  Lane changes: {self.lane_change_count[agent_idx]}")
            print(f"  Collisions: {self.collision_count[agent_idx]}")
            
            # Reset metrics for the next episode
            self.current_episode_reward[agent_idx] = 0.0
            self.current_episode_length[agent_idx] = 0
            self.current_episode_comfort[agent_idx] = []
            self.current_episode_efficiency[agent_idx] = []
            self.current_episode_safety[agent_idx] = []
            self.current_episode_traffic_flow[agent_idx] = []
            self.lane_change_count[agent_idx] = 0
            self.collision_count[agent_idx] = 0
            self.prev_lane[agent_idx] = None

    def calculate_system_performance(self, infos):
        """Calculate system-wide performance metrics based on all agent states"""
        # If we don't have info for multiple agents, return empty metrics
        if not isinstance(infos, dict) or len(infos) <= 1:
            return None
            
        # Extract key metrics from all agents
        # Safety metrics - based on collisions and safety rewards
        agent_safety_scores = []
        agent_efficiency_scores = []
        agent_distances = {}
        
        # Prepare to calculate coordination
        all_positions = {}
        all_speeds = {}
        all_lanes = {}
        
        # Collect data from all agents
        for agent_name, agent_info in infos.items():
            # Extract agent index
            try:
                agent_idx = int(agent_name.split('_')[1])
            except (ValueError, IndexError):
                continue  # Skip non-RL agent or invalid name
            
            # Collect safety scores
            safe_reward = float(agent_info.get("r_safe", 0.0))
            agent_safety_scores.append(safe_reward)
            
            # Collect efficiency scores
            eff_reward = float(agent_info.get("r_eff", 0.0))
            agent_efficiency_scores.append(eff_reward)
            
            # Collect positions and speeds for coordination calculation
            pos_x = agent_info.get("pos_x", None)
            pos_y = agent_info.get("pos_y", None)
            if pos_x is not None and pos_y is not None:
                all_positions[agent_name] = (pos_x, pos_y)
            
            speed = agent_info.get("speed", 0.0)
            all_speeds[agent_name] = speed
            
            lane = agent_info.get("curr_lane", "")
            all_lanes[agent_name] = lane
        
        # Calculate safety index - average of safety rewards (normalized to 0-1)
        if agent_safety_scores:
            avg_safety = sum(agent_safety_scores) / len(agent_safety_scores)
            # Convert from potential range of -1 to 1 to 0 to 1
            safety_index = (avg_safety + 1) / 2
        else:
            safety_index = 0.5  # Default if no data
            
        # Calculate efficiency index - average of efficiency rewards (normalized to 0-1)
        if agent_efficiency_scores:
            avg_efficiency = sum(agent_efficiency_scores) / len(agent_efficiency_scores)
            # Convert from potential range of -1 to 1 to 0 to 1
            efficiency_index = (avg_efficiency + 1) / 2
        else:
            efficiency_index = 0.5  # Default if no data
            
        # Calculate coordination index - based on relative positions, speeds, lane distribution
        # Higher coordination means agents maintain safe distances and similar speeds
        coordination_index = 0.5  # Default value
        
        # Simple coordination metric: variance in speeds (lower variance = higher coordination)
        if len(all_speeds) > 1:
            speeds = list(all_speeds.values())
            # Calculate speed variance
            mean_speed = sum(speeds) / len(speeds)
            speed_variance = sum((s - mean_speed) ** 2 for s in speeds) / len(speeds)
            
            # Convert variance to a coordination score (higher variance = lower coordination)
            # Using a simple exponential decay function: e^(-variance/10)
            coordination_index = min(1.0, max(0.0, np.exp(-speed_variance / 10)))
            
            # Further adjust based on lane distribution
            # If many agents are in the same lane, coordination is lower (traffic jam)
            lane_counts = {}
            for lane in all_lanes.values():
                if lane:
                    lane_counts[lane] = lane_counts.get(lane, 0) + 1
                    
            # If many agents in one lane, reduce coordination
            max_agents_in_lane = max(lane_counts.values()) if lane_counts else 0
            if max_agents_in_lane > len(all_lanes) / 2:
                coordination_index *= 0.8  # Penalty for lane crowding
        
        # Calculate overall system performance index
        # Weighted average of all metrics
        system_index = (
            0.3 * safety_index +
            0.4 * efficiency_index +
            0.3 * coordination_index
        )
        
        # Update running averages for system metrics
        self.current_safety_index = 0.9 * self.current_safety_index + 0.1 * safety_index
        self.current_efficiency_index = 0.9 * self.current_efficiency_index + 0.1 * efficiency_index
        self.current_coordination_index = 0.9 * self.current_coordination_index + 0.1 * coordination_index
        self.current_system_performance = 0.9 * self.current_system_performance + 0.1 * system_index
        
        self.system_metrics_counter += 1
        
        # Add metrics to our tracking lists - do this for every update for more data points
        self.safety_index.append(safety_index)
        self.efficiency_index.append(efficiency_index)
        self.coordination_index.append(coordination_index)
        self.system_performance_index.append(system_index)
        
        # Return for reporting
        return {
            "system_index": system_index,
            "safety": safety_index,
            "efficiency": efficiency_index,
            "coordination": coordination_index
        }

class SystemMetricsCallback(BaseCallback):
    """
    Custom callback to track system-wide metrics across all agents.
    """
    def __init__(self, metrics_callback, verbose=0):
        super().__init__(verbose)
        self.metrics_callback = metrics_callback
        
    def _on_step(self):
        # Extract infos from locals which contain all agent data
        if "infos" in self.locals and self.locals["infos"] and "all_agent_infos" in self.locals["infos"][0]:
            # We have info about all agents - pass to system metrics
            self.metrics_callback.calculate_system_performance(self.locals["infos"][0]["all_agent_infos"])
        return True

def make_multi_agent_env():
    """Create and initialize the multi-agent environment"""
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            print(f"Attempt {attempt+1}/{max_attempts} to create environment...")
            env = MultiAgentCarEnv(num_agents=NUM_AGENTS)
            
            # Start the environment with GUI to help debugging
            print("Starting multi-agent environment...")
            
            # Use fewer vehicles (10-15) for better visibility and reduced collision risk
            num_vehicles = np.random.randint(10, 15)
            print(f"Initializing with {num_vehicles} background vehicles")
            
            # Always use GUI during development/debugging
            success = env.start(gui=True, numVehicles=num_vehicles, vType='human')
            
            if not success:
                print(f"Failed to start SUMO environment on attempt {attempt+1}")
                if attempt < max_attempts - 1:
                    print("Waiting 3 seconds before next attempt...")
                    time.sleep(3)
                    continue
                raise RuntimeError("Failed to start SUMO environment after multiple attempts")
            
            # Test the environment with a few steps before returning
            print("Testing environment with a few simulation steps...")
            try:
                # Reset to get initial observations
                obs_dict, _ = env.reset()
                print(f"Reset successful, got observations for {len(obs_dict)} agents")
                
                # Take a few random steps to check stability
                for i in range(3):
                    actions = {}
                    for agent_id in env.agents:
                        # Random actions within valid bounds
                        actions[agent_id] = np.array([0.0, 0.0])  # Conservative actions for testing
                    
                    obs, rewards, terms, truncs, infos = env.step(actions)
                    print(f"Test step {i+1}: {len(env.agents)} active agents")
                
                print("Environment test successful!")
                return env
                
            except Exception as e:
                print(f"Error during environment test: {e}")
                env.close()
                if attempt < max_attempts - 1:
                    print("Waiting 3 seconds before next attempt...")
                    time.sleep(3)
                    continue
                raise RuntimeError(f"Environment test failed: {e}")
                
        except Exception as e:
            print(f"Error creating environment on attempt {attempt+1}: {e}")
            if attempt < max_attempts - 1:
                print("Waiting 3 seconds before next attempt...")
                time.sleep(3)
                continue
            raise
    
    raise RuntimeError("Failed to create environment after all attempts")

def train_multiagent_ppo():
    """Train all agents simultaneously using independent PPO"""
    # Create multi-agent environment
    print("Creating multi-agent environment...")
    marl_env = make_multi_agent_env()
    
    # Create callback for logging
    metrics_callback = MetricsCallback(num_agents=NUM_AGENTS)
    
    # Use the MultiPPOTrainer for truly simultaneous training
    print("Initializing MultiPPOTrainer for simultaneous training...")
    trainer = MultiPPOTrainer(marl_env, NUM_AGENTS, metrics_callback)
    
    # Create system metrics callback
    system_callback = SystemMetricsCallback(metrics_callback)
    
    # Train all agents simultaneously
    print(f"Starting simultaneous training of all {NUM_AGENTS} agents for {TOTAL_TIMESTEPS} total timesteps")
    trained_models = trainer.train(total_timesteps=TOTAL_TIMESTEPS, eval_freq=CHECKPOINT_FREQ)
    
    # Close environment
    marl_env.close()
    
    # Plot final results
    plot_training_results(metrics_callback)
    
    return metrics_callback

def plot_training_results(metrics_callback):
    """Plot and save training results for all agents"""
    plt.figure(figsize=(15, 12))
    
    # Plot rewards for all agents
    plt.subplot(2, 2, 1)
    for i in range(metrics_callback.num_agents):
        if metrics_callback.episode_rewards[i]:  # Only plot if we have data
            plt.plot(metrics_callback.episode_rewards[i], label=f'Agent {i}')
    plt.title('Training Rewards by Agent')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot episode lengths for all agents
    plt.subplot(2, 2, 2)
    for i in range(metrics_callback.num_agents):
        if metrics_callback.episode_lengths[i]:  # Only plot if we have data
            plt.plot(metrics_callback.episode_lengths[i], label=f'Agent {i}')
    plt.title('Episode Lengths by Agent')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True)
    
    # Plot lane changes for all agents
    plt.subplot(2, 2, 3)
    for i in range(metrics_callback.num_agents):
        if metrics_callback.episode_lane_changes[i]:  # Only plot if we have data
            plt.plot(metrics_callback.episode_lane_changes[i], label=f'Agent {i}')
    plt.title('Lane Changes per Episode by Agent')
    plt.xlabel('Episode')
    plt.ylabel('Number of Lane Changes')
    plt.legend()
    plt.grid(True)
    
    # Plot collisions for all agents 
    plt.subplot(2, 2, 4)
    for i in range(metrics_callback.num_agents):
        if metrics_callback.episode_collisions[i]:  # Only plot if we have data
            plt.plot(metrics_callback.episode_collisions[i], label=f'Agent {i}')
    plt.title('Collisions per Episode by Agent')
    plt.xlabel('Episode')
    plt.ylabel('Number of Collisions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", "all_agents_training.png"))
    plt.close()
    
    # Create a global performance overview plot
    create_global_performance_plot(metrics_callback)
    
    # Save metrics to file
    save_training_metrics(metrics_callback)

def create_global_performance_plot(metrics_callback):
    """Create a summary plot showing the overall performance metrics across all agents"""
    # Use timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(15, 10))
    
    # Calculate overall metrics
    agents_avg_rewards = []
    agents_avg_lengths = []
    agents_total_lane_changes = []
    agents_total_collisions = []
    agents_labels = []
    
    for i in range(metrics_callback.num_agents):
        if metrics_callback.episode_rewards[i]:  # Only include if we have data
            rewards = metrics_callback.episode_rewards[i]
            episode_lengths = metrics_callback.episode_lengths[i]
            
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            avg_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
            total_lane_changes = sum(metrics_callback.episode_lane_changes[i])
            total_collisions = sum(metrics_callback.episode_collisions[i])
            
            agents_avg_rewards.append(avg_reward)
            agents_avg_lengths.append(avg_length)
            agents_total_lane_changes.append(total_lane_changes)
            agents_total_collisions.append(total_collisions)
            agents_labels.append(f'Agent {i}')
    
    # Plot bar charts for comparison
    x = np.arange(len(agents_labels))
    width = 0.2
    
    # Plot 1: Average Rewards
    plt.subplot(2, 2, 1)
    plt.bar(x, agents_avg_rewards)
    plt.xlabel('Agent')
    plt.ylabel('Average Reward')
    plt.title('Average Reward by Agent')
    plt.xticks(x, agents_labels)
    plt.grid(True, axis='y')
    
    # Plot 2: Average Episode Lengths
    plt.subplot(2, 2, 2)
    plt.bar(x, agents_avg_lengths)
    plt.xlabel('Agent')
    plt.ylabel('Average Length')
    plt.title('Average Episode Length by Agent')
    plt.xticks(x, agents_labels)
    plt.grid(True, axis='y')
    
    # Plot 3: Total Lane Changes
    plt.subplot(2, 2, 3)
    plt.bar(x, agents_total_lane_changes)
    plt.xlabel('Agent')
    plt.ylabel('Total Lane Changes')
    plt.title('Total Lane Changes by Agent')
    plt.xticks(x, agents_labels)
    plt.grid(True, axis='y')
    
    # Plot 4: Total Collisions
    plt.subplot(2, 2, 4)
    plt.bar(x, agents_total_collisions)
    plt.xlabel('Agent')
    plt.ylabel('Total Collisions')
    plt.title('Total Collisions by Agent')
    plt.xticks(x, agents_labels)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", f"global_performance_summary_{timestamp}.png"))
    plt.close()
    
    # If we have system performance data, create a special plot
    if hasattr(metrics_callback, 'system_performance_index') and len(metrics_callback.system_performance_index) > 0:
        plt.figure(figsize=(15, 10))
        
        # Plot system performance index
        plt.subplot(2, 2, 1)
        plt.plot(metrics_callback.system_performance_index, label='System Performance', linewidth=2)
        plt.title('System Performance Index Over Training')
        plt.xlabel('Episode')
        plt.ylabel('Performance Index')
        plt.legend()
        plt.grid(True)
        
        # Plot safety index
        plt.subplot(2, 2, 2)
        plt.plot(metrics_callback.safety_index, label='Safety Index', linewidth=2, color='green')
        plt.title('Safety Index Over Training')
        plt.xlabel('Episode')
        plt.ylabel('Index Value (0-1)')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        # Plot efficiency index
        plt.subplot(2, 2, 3)
        plt.plot(metrics_callback.efficiency_index, label='Efficiency Index', linewidth=2, color='blue')
        plt.title('Efficiency Index Over Training')
        plt.xlabel('Episode')
        plt.ylabel('Index Value (0-1)')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        # Plot coordination index
        plt.subplot(2, 2, 4)
        plt.plot(metrics_callback.coordination_index, label='Coordination Index', linewidth=2, color='orange')
        plt.title('Coordination Index Over Training')
        plt.xlabel('Episode')
        plt.ylabel('Index Value (0-1)')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("graphs", f"system_performance_trend_{timestamp}.png"))
        plt.close()

def save_training_metrics(metrics_callback):
    """Save detailed training metrics to JSON file for later analysis"""
    # Prepare data structure for all metrics
    metrics = {
        "training_summary": {
            "num_agents": metrics_callback.num_agents,
            "total_episodes": sum(metrics_callback.n_episodes.values()),
            "total_steps": metrics_callback.global_step_count,
            "training_times": {i: metrics_callback.training_times[i] for i in range(metrics_callback.num_agents)},
        },
        "per_agent": {}
    }
    
    # Add per-agent metrics
    for i in range(metrics_callback.num_agents):
        if metrics_callback.episode_rewards[i]:  # Only include if we have data
            # Calculate statistics using native Python operations
            rewards = metrics_callback.episode_rewards[i]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            
            episode_lengths = metrics_callback.episode_lengths[i]
            avg_episode_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
            
            # Process component metrics
            comfort_avg = []
            efficiency_avg = []
            safety_avg = []
            traffic_flow_avg = []
            
            for comps in metrics_callback.episode_comfort[i]:
                comfort_avg.append(sum(comps) / len(comps) if comps else 0.0)
                
            for comps in metrics_callback.episode_efficiency[i]:
                efficiency_avg.append(sum(comps) / len(comps) if comps else 0.0)
                
            for comps in metrics_callback.episode_safety[i]:
                safety_avg.append(sum(comps) / len(comps) if comps else 0.0)
                
            for comps in metrics_callback.episode_traffic_flow[i]:
                traffic_flow_avg.append(sum(comps) / len(comps) if comps else 0.0)
            
            agent_data = {
                "episodes": metrics_callback.n_episodes[i],
                "rewards": rewards,
                "episode_lengths": episode_lengths,
                "lane_changes": metrics_callback.episode_lane_changes[i],
                "collisions": metrics_callback.episode_collisions[i],
                "avg_reward": avg_reward,
                "avg_episode_length": avg_episode_length,
                "total_lane_changes": sum(metrics_callback.episode_lane_changes[i]),
                "total_collisions": sum(metrics_callback.episode_collisions[i]),
                "reward_components": {
                    "comfort": comfort_avg,
                    "efficiency": efficiency_avg,
                    "safety": safety_avg,
                    "traffic_flow": traffic_flow_avg
                }
            }
            metrics["per_agent"][f"agent_{i}"] = agent_data
    
    # Add global metrics
    if metrics_callback.global_rewards:
        global_rewards = metrics_callback.global_rewards
        global_lane_changes = metrics_callback.global_lane_changes
        global_collisions = metrics_callback.global_collisions
        
        metrics["global"] = {
            "rewards": global_rewards,
            "lane_changes": global_lane_changes,
            "collisions": global_collisions,
            "avg_reward": sum(global_rewards) / len(global_rewards) if global_rewards else 0.0,
            "avg_lane_changes": sum(global_lane_changes) / len(global_lane_changes) if global_lane_changes else 0.0,
            "collision_rate": sum([1 if c > 0 else 0 for c in global_collisions]) / len(global_collisions) if global_collisions else 0.0
        }
    
    # Add system performance metrics if available
    if hasattr(metrics_callback, 'system_performance_index') and metrics_callback.system_performance_index:
        system_index = metrics_callback.system_performance_index
        safety_index = metrics_callback.safety_index
        efficiency_index = metrics_callback.efficiency_index
        coordination_index = metrics_callback.coordination_index
        
        metrics["system_performance"] = {
            "overall_index": system_index,
            "safety_index": safety_index,
            "efficiency_index": efficiency_index,
            "coordination_index": coordination_index,
            "avg_performance": sum(system_index) / len(system_index) if system_index else 0.0,
            "avg_safety": sum(safety_index) / len(safety_index) if safety_index else 0.0,
            "avg_efficiency": sum(efficiency_index) / len(efficiency_index) if efficiency_index else 0.0,
            "avg_coordination": sum(coordination_index) / len(coordination_index) if coordination_index else 0.0
        }
    
    # Save to file with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_path = os.path.join("logs", f"training_metrics_{timestamp}.json")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training metrics saved to {metrics_path}")
    
    # Also save a summary text file with key metrics
    summary_path = os.path.join("logs", f"training_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("===== MULTI-AGENT REINFORCEMENT LEARNING TRAINING SUMMARY =====\n\n")
        f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of agents: {metrics_callback.num_agents}\n")
        f.write(f"Total training steps: {metrics_callback.global_step_count}\n\n")
        
        f.write("Per-Agent Performance:\n")
        f.write("-----------------------\n")
        for i in range(metrics_callback.num_agents):
            if metrics_callback.episode_rewards[i]:
                avg_reward = sum(metrics_callback.episode_rewards[i]) / len(metrics_callback.episode_rewards[i])
                total_episodes = metrics_callback.n_episodes[i]
                training_time = metrics_callback.training_times[i]
                time_str = f"{training_time:.1f}s" if training_time < 60 else f"{training_time/60:.1f}m"
                
                f.write(f"Agent {i}:\n")
                f.write(f"  Episodes completed: {total_episodes}\n")
                f.write(f"  Average reward: {avg_reward:.2f}\n")
                f.write(f"  Total lane changes: {sum(metrics_callback.episode_lane_changes[i])}\n")
                f.write(f"  Total collisions: {sum(metrics_callback.episode_collisions[i])}\n")
                f.write(f"  Training time: {time_str}\n\n")
        
        # Add system performance summary if available
        if hasattr(metrics_callback, 'system_performance_index') and metrics_callback.system_performance_index:
            system_index = metrics_callback.system_performance_index
            safety_index = metrics_callback.safety_index
            efficiency_index = metrics_callback.efficiency_index
            coordination_index = metrics_callback.coordination_index
            
            f.write("\nSystem Performance Metrics:\n")
            f.write("--------------------------\n")
            f.write(f"Overall System Performance Index: {sum(system_index)/len(system_index) if system_index else 0.0:.2f}\n")
            f.write(f"Safety Index: {sum(safety_index)/len(safety_index) if safety_index else 0.0:.2f}\n")
            f.write(f"Efficiency Index: {sum(efficiency_index)/len(efficiency_index) if efficiency_index else 0.0:.2f}\n")
            f.write(f"Coordination Index: {sum(coordination_index)/len(coordination_index) if coordination_index else 0.0:.2f}\n\n")
    
    print(f"Training summary saved to {summary_path}")

# MARL wrapper for parallel training
class MARLWrapper(gym.Env):
    """
    Wraps the multi-agent environment to provide a single-agent view for a specific agent,
    while other agents also take actions using their policies.
    """
    def __init__(self, env, agent_id, other_models=None):
        self.env = env
        self.agent_id = agent_id
        self.other_models = other_models or {}  # Dict mapping agent_id -> model
        self.last_observations = {}  # Track latest observations for all agents
        
        # Define the observation and action spaces for this agent
        self.observation_space = env.observation_spaces[agent_id]
        self.action_space = env.action_spaces[agent_id]
        
        # Required for Gymnasium Env
        self.metadata = {"render_modes": ["human"]}
    
    def reset(self, **kwargs):
        obs_dict, info_dict = self.env.reset(**kwargs)
        # Store last observations for all agents
        self.last_observations = obs_dict.copy()
        
        # Get observation for this agent
        obs = obs_dict.get(self.agent_id, np.zeros(self.observation_space.shape, dtype=np.float32))
        info = info_dict.get(self.agent_id, {})
        # Add system info
        info["all_agent_infos"] = info_dict
        return obs, info
    
    def step(self, action):
        # Create action dict for all agents
        actions = {self.agent_id: action}
        
        # Get actions from other trained models if available
        for other_id in self.env.agents:
            if other_id != self.agent_id:
                if other_id in self.other_models and other_id in self.env.agents:
                    # Get observation for this agent safely
                    other_obs = self._get_observation_safely(other_id)
                    
                    # Get action from the agent's policy
                    try:
                        other_action, _ = self.other_models[other_id].predict(other_obs, deterministic=True)
                        actions[other_id] = other_action
                    except Exception as e:
                        actions[other_id] = self._get_fallback_action(other_id, e)
                else:
                    # Use rule-based action for agents without models
                    actions[other_id] = self._get_rule_based_action(other_id)
        
        # Step the environment
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(actions)
        
        # Update last observations
        self.last_observations = obs_dict.copy()
        
        # Get results for this agent
        obs = obs_dict.get(self.agent_id, np.zeros(self.observation_space.shape, dtype=np.float32))
        rew = rew_dict.get(self.agent_id, 0.0)
        term = term_dict.get(self.agent_id, False)
        trunc = trunc_dict.get(self.agent_id, False)
        info = info_dict.get(self.agent_id, {})
        
        # Add system info
        info["all_agent_infos"] = info_dict
        
        return obs, rew, term, trunc, info
    
    def render(self):
        # Pass through to env's render method if it exists
        if hasattr(self.env, 'render'):
            return self.env.render()
    
    def close(self):
        # Pass through to env's close method if it exists
        if hasattr(self.env, 'close'):
            return self.env.close()
            
    def _get_observation_safely(self, agent_id):
        """Safely retrieve observation for an agent with appropriate error handling"""
        # Prioritize getting observation from last_observations
        if agent_id in self.last_observations:
            return self.last_observations[agent_id]
        
        # Try different methods to get a valid observation
        try:
            # Try get_state method if available
            if hasattr(self.env, 'get_state'):
                return self.env.get_state(agent_name=agent_id)
            
            # Try _get_obs method if available
            if hasattr(self.env, '_get_obs'):
                obs_dict = self.env._get_obs()
                if agent_id in obs_dict:
                    return obs_dict[agent_id]
                
            # Try get_observation method if available  
            if hasattr(self.env, 'get_observation'):
                return self.env.get_observation(agent_id)
        except Exception as e:
            print(f"Error getting observation for agent {agent_id}: {str(e)[:50]}...")
        
        # Return zeros as a last resort
        return np.zeros(self.env.observation_spaces[agent_id].shape, dtype=np.float32)
    
    def _get_fallback_action(self, agent_id, error=None):
        """Generate a safe fallback action when model prediction fails"""
        if error:
            print(f"Using fallback action for {agent_id} due to: {str(error)[:50]}...")
            
        # Create zeros but with small random noise for exploration
        action_space = self.env.action_spaces[agent_id]
        action = np.zeros(action_space.shape, dtype=np.float32)
        
        # Add small random noise for minimal exploration
        action += np.random.normal(0, 0.1, size=action_space.shape)
        
        # Clip to valid range
        action = np.clip(action, action_space.low, action_space.high)
        
        return action
    
    def _get_rule_based_action(self, agent_id):
        """Generate a rule-based action for agents without models"""
        action_space = self.env.action_spaces[agent_id]
        rule_action = np.zeros(action_space.shape, dtype=np.float32)
        
        try:
            # Try to implement simple driving behavior: maintain target speed
            current_speed = getattr(self.env, 'speed', {}).get(agent_id, 0)
            target_speed = getattr(self.env, 'target_speed', {}).get(agent_id, 10)
            speed_diff = target_speed - current_speed
            
            # First action component is usually acceleration
            if action_space.shape[0] > 0:
                rule_action[0] = np.clip(speed_diff / 4.0, -1.0, 1.0)
            
            # Second component is usually lane change
            if action_space.shape[0] > 1:
                lane_change_cooldown = getattr(self.env, 'lane_change_cooldown', {}).get(agent_id, 0)
                if lane_change_cooldown == 0 and np.random.random() < 0.05:
                    rule_action[1] = np.random.choice([-0.7, 0.7])
        except Exception:
            # If anything fails, just return zeros
            pass
        
        return rule_action

if __name__ == "__main__":
    try:
        print("Starting Multi-Agent Reinforcement Learning with Independent PPO...")
        metrics = train_multiagent_ppo()  # Use the new PPO training function
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc() 