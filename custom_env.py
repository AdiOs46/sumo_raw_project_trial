import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import numpy as np
from scipy.spatial import distance
from math import atan2, degrees
from collections import deque
import sumolib
import sys
import os
import time
import subprocess
import atexit
import signal
import logging

def angle_between(p1, p2, rl_angle):
	xDiff = p2[0] - p1[0]
	yDiff = p2[1] - p1[1]
	angle = degrees(atan2(yDiff, xDiff))
	# Adding the rotation angle of the agent
	angle += rl_angle
	angle = angle % 360
	return angle

def get_distance(a, b):
	return distance.euclidean(a, b)

class MultiAgentCarEnv(gym.Env):
    def __init__(self):
        self.name = 'rlagent'
        self.step_length = 0.4
        self.acc_history = deque([0, 0], maxlen=2)
        self.grid_state_dim = 3
        self.grid_cells = self.grid_state_dim * self.grid_state_dim  # 9 cells
        self.features_per_vehicle = 4  # [dist/pos_x, pos_y/speed, speed/acc, lat_speed/lat_pos]
        self.agent_features = 5  # [pos_x, pos_y, speed, lat_speed, acc]
        self.state_dim = (self.grid_cells * self.features_per_vehicle) + 1
        
        # Validate state dimension
        assert self.state_dim == 37, f"Expected state dimension 37, got {self.state_dim}"
        
        self.pos = (0, 0)
        self.curr_lane = ''
        self.curr_sublane = -1
        self.prev_sublane = -1  # Track previous lane for lane change detection
        self.target_speed = 0
        self.speed = 0
        self.lat_speed = 0
        self.acc = 0
        self.angle = 0
        self.gui = True
        self.numVehicles = 0
        self.vType = 0
        self.lane_ids = []
        self.max_steps = 10000
        self.curr_step = 0
        self.collision = False
        self.done = False
        self.traci_started = False
        self.sumo_process = None
        self.lane_change_attempted = False  # Track if lane change was attempted
        self.lane_change_cooldown = 0       # Cooldown counter for lane changes
        self.lane_change_cooldown_steps = 10  # Minimum steps between lane changes (4 seconds at 0.4s step)
        
        # Register an exit handler to ensure SUMO closes on program exit
        atexit.register(self.close_on_exit)

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,),  # Adjust the shape based on your observation vector size
            dtype=np.float32
        )

        # Define action space
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),  # Adjust the shape based on your action vector size
            dtype=np.float32
        )
        
        self.logger = logging.getLogger("MultiAgentCarEnv")
        self.logger.setLevel(logging.INFO)
    
    def close_on_exit(self):
        self.close()
        
    def start(self, gui=True, numVehicles=30, vType='human', network_conf="config/config.sumocfg", network_xml='config/network.net.xml'):
        # Close any existing connections
        self.close()
        
        # Wait a bit to ensure previous process terminated
        time.sleep(1.0)
        
        self.gui = gui
        self.numVehicles = numVehicles
        self.vType = vType
        self.network_conf = network_conf
        
        # Check if network files exist
        if not os.path.exists(network_xml):
            raise FileNotFoundError(f"Network file not found: {network_xml}")
        if not os.path.exists(network_conf):
            raise FileNotFoundError(f"Config file not found: {network_conf}")
            
        self.net = sumolib.net.readNet(network_xml)
        self.curr_step = 0
        self.collision = False
        self.done = False
    
        # Starting SUMO with appropriate parameters
        if gui:
            sumoCmd = ["sumo-gui", "-c", self.network_conf, 
                      "--no-step-log", "true", 
                      "--start",
                      "--quit-on-end",  # Allow SUMO to quit when closed
                      "--collision.action", "warn",  # Important: don't remove vehicles on collision
                      "--collision.mingap-factor", "0",  # Allow collisions to be detected
                      "--step-length", str(self.step_length)]
        else:
            # Use headless SUMO for faster training
            sumoCmd = ["sumo", "-c", self.network_conf, 
                      "--no-step-log", "true",
                      "--quit-on-end",
                      "--collision.action", "warn",
                      "--collision.mingap-factor", "0",
                      "--step-length", str(self.step_length)]
        
        try:
            # Start SUMO process using subprocess to better manage lifecycle
            if self.gui:
                # For GUI mode, let TraCI handle the connection
                traci.start(sumoCmd)
            else:
                # For headless mode, use subprocess for better process management
                self.sumo_process = subprocess.Popen(sumoCmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                traci.init(port=traci.getFreeSocketPort())
                
            self.traci_started = True
            
            # Get lane IDs if possible, with error handling
            try:
                self.lane_ids = traci.lane.getIDList()
            except traci.exceptions.FatalTraCIError:
                print("TraCI connection failed. Retrying...")
                return False
            
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            return False
    
        # Populating the highway
        vehicles_added = 0
        for i in range(self.numVehicles):
            veh_name = 'vehicle_' + str(i)
            try:
                traci.vehicle.add(veh_name, routeID='route_0', typeID=self.vType, departLane='random')
                # Make vehicles continue driving by setting "keepRoute" mode
                traci.vehicle.setLaneChangeMode(veh_name, 256)
                vehicles_added += 1
            except traci.exceptions.TraCIException as e:
                print(f"Error adding vehicle {veh_name}: {e}")
                
        if vehicles_added == 0:
            print("Failed to add any vehicles. Check your route definitions.")
            return False
        
        # Add the RL agent
        try:
            traci.vehicle.add(self.name, routeID='route_0', typeID='rl')
            # Setting the lane change mode for the RL agent - 0 means full control is given to TraCI
            traci.vehicle.setLaneChangeMode(self.name, 0)
        except traci.exceptions.TraCIException as e:
            print(f"Error adding RL agent: {e}")
            return False
    
        # Do some random steps to distribute the vehicles
        steps_completed = 0
        for step in range(min(self.numVehicles*4, 100)):  # Limit to reasonable number
            try:
                traci.simulationStep()
                steps_completed += 1
                
                # Check if SUMO is still running
                if not self.is_sumo_running():
                    print("SUMO was closed during initialization")
                    return False
                    
            except Exception as e:
                print(f"Error during initial simulation steps: {e}")
                return False
        
        if steps_completed == 0:
            print("Failed to complete any initialization steps.")
            return False
    
        # Setting up useful parameters
        try:
            self.update_params()
            return True
        except Exception as e:
            print(f"Error updating parameters: {e}")
            return False
    
    def is_sumo_running(self):
        """Check if SUMO is still running"""
        try:
            # Check if we can still get simulation time - will raise exception if connection is gone
            traci.simulation.getTime()
            return True
        except traci.exceptions.FatalTraCIError:
            return False
        except:
            return False
        
    def update_params(self):
        # initialize params
        try:
            # First check if SUMO is still running
            if not self.is_sumo_running():
                self.traci_started = False
                return
                
            # Now try to update parameters
            if self.name not in traci.vehicle.getIDList():
                # If the RL agent is not in the simulation, try to add it back
                try:
                    traci.vehicle.add(self.name, routeID='route_0', typeID='rl')
                    traci.vehicle.setLaneChangeMode(self.name, 0)
                    # Give SUMO a step to register the vehicle
                    traci.simulationStep()
                except:
                    pass
            
            # Get position and lane safely
            try:
                self.pos = traci.vehicle.getPosition(self.name)
                self.curr_lane = traci.vehicle.getLaneID(self.name)
            except:
                # Default values if vehicle doesn't exist
                self.pos = (0, 0)
                self.curr_lane = "e_0"  # Default lane
            
            if not self.curr_lane:
                # Handle case when the vehicle is in teleport state
                if self.name in traci.simulation.getStartingTeleportIDList():
                    # Wait for teleport to finish
                    teleport_waiting_steps = 0
                    while self.name in traci.simulation.getStartingTeleportIDList() and teleport_waiting_steps < 10:
                        traci.simulationStep()
                        teleport_waiting_steps += 1
                    try:
                        self.curr_lane = traci.vehicle.getLaneID(self.name)
                    except:
                        self.curr_lane = "e_0"  # Default if still issues
                else:
                    # Vehicle might be removed, try to add it back
                    try:
                        traci.vehicle.add(self.name, routeID='route_0', typeID='rl')
                        traci.vehicle.setLaneChangeMode(self.name, 0)
                        traci.simulationStep()
                        self.curr_lane = traci.vehicle.getLaneID(self.name)
                    except:
                        # If we can't add it back, use default values
                        self.curr_lane = "e_0"  # Default lane
            
            # Parse sublane index from lane ID
            try:
                if '_' in self.curr_lane:
                    self.curr_sublane = int(self.curr_lane.split("_")[1])
                else:
                    self.curr_sublane = 0  # Default value
            except (IndexError, ValueError):
                self.curr_sublane = 0  # Default value
            
            # Get other vehicle parameters safely with defaults
            try:
                self.target_speed = traci.vehicle.getAllowedSpeed(self.name)
            except:
                self.target_speed = 13.89  # Default 50km/h in m/s
                
            try:
                self.speed = traci.vehicle.getSpeed(self.name)
            except:
                self.speed = 0
                
            try:
                self.lat_speed = traci.vehicle.getLateralSpeed(self.name)
            except:
                self.lat_speed = 0
                
            try:
                self.acc = traci.vehicle.getAcceleration(self.name)
            except:
                self.acc = 0
                
            self.acc_history.append(self.acc)
            
            try:
                self.angle = traci.vehicle.getAngle(self.name)
            except:
                self.angle = 0
                
        except traci.exceptions.TraCIException as e:
            print(f"Error in update_params: {e}")
            # Use default values if there's an error
            if len(self.acc_history) < 2:
                self.acc_history.append(0)
            if not hasattr(self, 'pos') or not self.pos:
                self.pos = (0, 0)

    def get_grid_state(self, threshold_distance=10):
        '''
        Observation is a grid occupancy grid
		'''
        agent_lane = self.curr_lane
        agent_pos = self.pos
        
        # Default to edge "e" if curr_lane is invalid
        if not self.curr_lane or not "_" in self.curr_lane:
            edge = "e"
        else:
            edge = self.curr_lane.split("_")[0]
            
        agent_lane_index = self.curr_sublane
        
        # Get lanes safely
        try:
            lanes = [lane for lane in self.lane_ids if edge in lane]
        except:
            lanes = []  # If we can't get lanes, use empty list
            
        state = np.zeros([self.grid_state_dim, self.grid_state_dim])
        
        # Putting agent
        agent_x, agent_y = 1, agent_lane_index
        # Ensure agent_y is within bounds
        agent_y = max(0, min(agent_y, self.grid_state_dim-1))
        state[agent_x, agent_y] = -1
        
        # Put other vehicles
        for lane in lanes:
            # Get vehicles in the lane
            try:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
            except traci.exceptions.TraCIException:
                continue
                
            try:
                veh_lane = int(lane.split("_")[-1])
            except (IndexError, ValueError):
                continue
                
            for vehicle in vehicles:
                if vehicle == self.name:
                    continue
                    
                try:
                    # Get angle wrt rlagent
                    veh_pos = traci.vehicle.getPosition(vehicle)
                    # If too far, continue
                    if get_distance(agent_pos, veh_pos) > threshold_distance:
                        continue
                        
                    rl_angle = traci.vehicle.getAngle(self.name)
                    # Extract vehicle ID safely
                    try:
                        veh_id = int(vehicle.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                        
                    angle = angle_between(agent_pos, veh_pos, rl_angle)
                    
                    # Ensure veh_lane is within bounds
                    veh_lane = max(0, min(veh_lane, self.grid_state_dim-1))
                    
                    # Putting on the right
                    if angle > 337.5 or angle < 22.5:
                        state[agent_x, veh_lane] = veh_id
                    # Putting on the right north
                    elif angle >= 22.5 and angle < 67.5:
                        state[max(0, agent_x-1), veh_lane] = veh_id
                    # Putting on north
                    elif angle >= 67.5 and angle < 112.5:
                        state[max(0, agent_x-1), veh_lane] = veh_id
                    # Putting on the left north
                    elif angle >= 112.5 and angle < 157.5:
                        state[max(0, agent_x-1), veh_lane] = veh_id
                    # Putting on the left
                    elif angle >= 157.5 and angle < 202.5:
                        state[agent_x, veh_lane] = veh_id
                    # Putting on the left south
                    elif angle >= 202.5 and angle < 237.5:
                        state[min(self.grid_state_dim-1, agent_x+1), veh_lane] = veh_id
                    # Putting on the south
                    elif angle >= 237.5 and angle < 292.5:
                        state[min(self.grid_state_dim-1, agent_x+1), veh_lane] = veh_id
                    # Putting on the right south
                    elif angle >= 292.5 and angle < 337.5:
                        state[min(self.grid_state_dim-1, agent_x+1), veh_lane] = veh_id
                except Exception as e:
                    continue
                    
        # Since the 0 lane is the right most one, flip 
        state = np.fliplr(state)
        return state
    
    def render(self, mode='human', close=False):
        pass
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        Args:
            seed: The seed for randomization
            options: Additional options for environment reset
        Returns:
            observation: The initial observation
            info: Additional information
        """
        # Handle the seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset lane change tracking
        self.prev_sublane = -1
        self.lane_change_attempted = False
        self.lane_change_cooldown = 0  # Reset cooldown counter
        
        # Get options with defaults
        if options is None:
            options = {}
        gui = options.get('gui', True)  # Default to GUI=True to help with debugging
        numVehicles = options.get('numVehicles', np.random.randint(15, 40))
        vType = options.get('vType', 'human')
        
        # Close any existing simulation first
        self.close()
        
        # Try up to 3 times to start the simulation
        for attempt in range(3):
            success = self.start(gui, numVehicles, vType)
            if success:
                break
            print(f"Attempt {attempt+1} failed. Retrying...")
            time.sleep(1)
            
        if not success:
            raise RuntimeError("Failed to start SUMO simulation after multiple attempts")
                
        try:
            initial_state = self.get_state()
        except Exception as e:
            print(f"Error getting initial state: {e}")
            # Return a zero state if there's an error
            initial_state = np.zeros(self.state_dim)
        
        # Return both the initial state and an empty info dict
        return initial_state, {}
    
    def close(self):
        """Clean up TraCI and terminate SUMO process"""
        if self.traci_started:
            try:
                # Close TraCI connection
                traci.close()
            except:
                pass
                
            self.traci_started = False
            
        # Kill SUMO process if it exists
        if self.sumo_process is not None:
            try:
                self.sumo_process.terminate()
                self.sumo_process.wait(timeout=2)
            except:
                # Force kill if terminate doesn't work
                try:
                    self.sumo_process.kill()
                except:
                    pass
                    
            self.sumo_process = None
            
        # Give system time to clean up ports, etc.
        time.sleep(0.5)
    
    def compute_jerk(self):
        if len(self.acc_history) < 2:
            return 0
        return (self.acc_history[1] - self.acc_history[0])/self.step_length
    
    def detect_collision(self):
        try:
            collisions = traci.simulation.getCollidingVehiclesIDList()
            if self.name in collisions:
                # Get collision details if possible
                try:
                    collision_vehicles = []
                    for veh in collisions:
                        if veh != self.name and veh in traci.vehicle.getIDList():
                            # Get the colliding vehicle's info
                            veh_speed = traci.vehicle.getSpeed(veh)
                            veh_lane = traci.vehicle.getLaneID(veh)
                            veh_pos = traci.vehicle.getPosition(veh)
                            dist = get_distance(self.pos, veh_pos)
                            collision_vehicles.append(f"{veh} (speed={veh_speed:.1f}, lane={veh_lane}, dist={dist:.1f})")
                    
                    # Log detailed collision info
                    print(f"COLLISION DETECTED: Agent speed={self.speed:.1f}, lane={self.curr_lane}")
                    print(f"Collided with: {', '.join(collision_vehicles) if collision_vehicles else 'Unknown'}")
                except:
                    print("COLLISION DETECTED: Could not get detailed collision info")
                
                self.collision = True
                return True
        except:
            # If we can't get collision info, assume no collision
            pass
            
        self.collision = False
        return False

    def step(self, action):
        # Actions: continuous action space: action[0] for acceleration, action[1] for lane change

        # Store current lane to detect changes later
        self.prev_sublane = self.curr_sublane
        # Reset lane change tracking
        self.lane_change_attempted = False
        
        # Update lane change cooldown counter
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

        # First check if SUMO is still running
        if not self.is_sumo_running():
            print("SUMO has closed unexpectedly. Attempting to restart...")
            self.traci_started = False
            success = self.start(self.gui, self.numVehicles, self.vType)
            if not success:
                # If restart fails, return terminal state
                print("Failed to restart. Ending episode.")
                zero_state = np.zeros(self.state_dim)
                return zero_state, -10, True, True, {"error": "SUMO disconnected"}

        # Process acceleration control (if using continuous action space)
        if isinstance(action, (list, np.ndarray)) and len(action) >= 1:
            try:
                # Scale the action to a reasonable acceleration range (-4 to 4 m/s²)
                acceleration = action[0] * 4.0
                
                # Get current speed
                current_speed = traci.vehicle.getSpeed(self.name)
                
                # Calculate target speed (current + acceleration * timestep)
                target_speed = max(0, current_speed + acceleration * self.step_length)
                
                # Apply the speed change
                traci.vehicle.setSpeed(self.name, target_speed)
            except:
                pass  # Silently ignore speed setting errors

        # Get current lane information
        try:
            if self.curr_lane and '_' in self.curr_lane:
                edge = self.curr_lane.split("_")[0]
                lane_idx = self.curr_sublane
                lanes_on_edge = [lane for lane in self.lane_ids if edge in lane]
                num_lanes = len(lanes_on_edge)
            else:
                # Default values if lane information is invalid
                edge = "e"
                lane_idx = 0
                lanes_on_edge = [lane for lane in self.lane_ids if edge in lane]
                num_lanes = len(lanes_on_edge) if lanes_on_edge else 1
        except:
            edge = "e"
            lane_idx = 0
            num_lanes = 1
        
        # Process lane change with improved implementation and cooldown
        if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
            lane_change_action = action[1]
            
            try:
                # Only attempt lane changes if cooldown has expired
                if self.lane_change_cooldown == 0 and self.name in traci.vehicle.getIDList():
                    # Get current lane information
                    current_lane = traci.vehicle.getLaneID(self.name)
                    if current_lane and '_' in current_lane:
                        edge = current_lane.split("_")[0]
                        current_lane_idx = self.curr_sublane
                        lanes_on_edge = [lane for lane in self.lane_ids if edge in lane]
                        num_lanes = len(lanes_on_edge)
                        
                        # Make lane change decisions more continuous with gradual thresholds
                        # Use a more gradual transition instead of sharp step function
                        # Need stronger action to initiate lane change (increased thresholds)
                        if lane_change_action < -0.5 and current_lane_idx > 0:  # Changed from -0.3 to -0.5
                            target_lane_idx = current_lane_idx - 1
                            # Use appropriate SUMO lane change reason (0=strategic change)
                            reason = 0
                            self.lane_change_attempted = True
                        elif lane_change_action > 0.5 and current_lane_idx < num_lanes - 1:  # Changed from 0.3 to 0.5
                            target_lane_idx = current_lane_idx + 1
                            reason = 0  # Strategic lane change
                            self.lane_change_attempted = True
                        else:
                            target_lane_idx = current_lane_idx
                            reason = 0  # Stay in lane
                        
                        # Check if lane change is safe
                        if target_lane_idx != current_lane_idx:
                            target_lane = lanes_on_edge[target_lane_idx]
                            vehicles_in_target = traci.lane.getLastStepVehicleIDs(target_lane)
                            
                            # Check for vehicles in target lane
                            safe_to_change = True
                            for veh in vehicles_in_target:
                                try:
                                    veh_pos = traci.vehicle.getPosition(veh)
                                    veh_speed = traci.vehicle.getSpeed(veh)
                                    dist = get_distance(self.pos, veh_pos)
                                    
                                    # More strict safety check (increased safety distances)
                                    if dist < 15 or (veh_speed > self.speed and dist < 25):  # Changed from 10/20 to 15/25
                                        safe_to_change = False
                                        break
                                except:
                                    continue
                            
                            if safe_to_change:
                                try:
                                    # Attempt lane change with reason
                                    traci.vehicle.changeLane(self.name, target_lane_idx, 0)  # Using 0 as strategic reason
                                    
                                    # Set cooldown after lane change attempt
                                    self.lane_change_cooldown = self.lane_change_cooldown_steps
                                    print(f"Lane change attempted. Cooldown set for {self.lane_change_cooldown} steps.")
                                except Exception as e:
                                    print(f"Lane change error: {e}")
            except Exception as e:
                print(f"Lane change processing error: {e}")
                pass

        # Route recycling logic for all vehicles
        try:
            veh_list = traci.vehicle.getIDList()
            
            # Recycle other vehicles whose remaining distance is very low
            for veh in veh_list:
                try:
                    # Check if vehicle has route info
                    route_edges = traci.vehicle.getRoute(veh)
                    if not route_edges:
                        continue
                        
                    # Get remaining distance if possible
                    try:
                        rem_distance = traci.vehicle.getRemainingDistance(veh)
                    except:
                        # If can't get remaining distance, check position on route
                        try:
                            route_idx = traci.vehicle.getRouteIndex(veh)
                            route_len = len(route_edges)
                            if route_idx >= route_len - 1:  # Near end of route
                                # Reset to beginning of route
                                traci.vehicle.setRoute(veh, route_edges)
                        except:
                            pass
                        continue
                        
                    if rem_distance is not None and rem_distance < 50:
                        # Reset to beginning of route
                        traci.vehicle.setRoute(veh, route_edges)
                except:
                    continue

            # Handle the RL agent specifically
            if self.name not in veh_list:
                try:
                    # Try to add the RL agent back
                    traci.vehicle.add(self.name, routeID='route_0', typeID='rl')
                    traci.vehicle.setLaneChangeMode(self.name, 0)
                except:
                    pass
        except:
            pass  # Silently ignore recycling errors

        # Advance simulation step
        try:
            traci.simulationStep()
        except:
            # If simulation step fails, SUMO likely crashed or was closed
            self.traci_started = False
            print("Simulation step failed. SUMO may have been closed.")
            return np.zeros(self.state_dim), -10, True, True, {"error": "simulation_step_failed"}

        # Check if SUMO is still running after step
        if not self.is_sumo_running():
            self.traci_started = False
            print("SUMO closed during simulation step")
            return np.zeros(self.state_dim), -10, True, True, {"error": "SUMO disconnected after step"}

        # Check collision
        collision = self.detect_collision()

        # Compute reward
        try:
            reward_total, r_comf, r_eff, r_safe, r_traffic = self.compute_reward(collision, action)
        except Exception as e:
            print(f"Error computing reward: {e}")
            reward_total, r_comf, r_eff, r_safe, r_traffic = -1, 0, 0, -1, 0

        # Update agent parameters from SUMO
        try:
            self.update_params()
        except Exception as e:
            print(f"Error updating parameters: {e}")

        # Get next observation state
        try:
            next_state = self.get_state()
        except Exception as e:
            print(f"Error getting next state: {e}")
            next_state = np.zeros(self.state_dim)

        # Update the step counter
        self.curr_step += 1

        # Determine if the episode should end
        done = collision  # End episode on collision
        truncated = self.curr_step >= self.max_steps

        info = {
            "r_comf": r_comf,
            "r_eff": r_eff,
            "r_safe": r_safe,
            "r_traffic": r_traffic,
            "collision": collision,
            "curr_lane": self.curr_lane,
            "curr_sublane": self.curr_sublane,
            "speed": self.speed,
            "steps": self.curr_step,
            "lane_change_cooldown": self.lane_change_cooldown
        }
        
        return next_state, reward_total, done, truncated, info

    def get_vehicle_info(self, vehicle_name):
        # Method to populate the vector information of a vehicle
        try:
            if vehicle_name == self.name:
                return np.array([self.pos[0], self.pos[1], self.speed, self.lat_speed, self.acc])
            else:
                # Check if vehicle exists
                if vehicle_name not in traci.vehicle.getIDList():
                    return np.zeros(4)  # Return zeros if vehicle doesn't exist
                    
                lat_pos, long_pos = traci.vehicle.getPosition(vehicle_name)
                long_speed = traci.vehicle.getSpeed(vehicle_name)
                acc = traci.vehicle.getAcceleration(vehicle_name)
                dist = get_distance(self.pos, (lat_pos, long_pos))
                return np.array([dist, long_speed, acc, lat_pos])
        except Exception as e:
            # Return zeros on error
            if vehicle_name == self.name:
                return np.zeros(5)
            return np.zeros(4)
    
    def get_state(self):
        # Define a state as a vector of vehicles information
        state = np.zeros(self.state_dim)
        
        try:
            before = 0
            grid_state = self.get_grid_state().flatten()
            
            for num, vehicle in enumerate(grid_state):
                if vehicle == 0:
                    continue
                if vehicle == -1:
                    vehicle_name = self.name
                    before = 1
                else:
                    vehicle_name = f'vehicle_{int(vehicle)}'
                
                veh_info = self.get_vehicle_info(vehicle_name)
                
                # Make sure we don't go out of bounds
                idx_init = num * 4
                if before and vehicle != -1:
                    idx_init += 1
                    
                idx_fin = min(idx_init + veh_info.shape[0], self.state_dim)
                state[idx_init:idx_fin] = veh_info[:idx_fin-idx_init]
                
            state = np.squeeze(state)
            
            # Ensure the state has the correct shape
            if state.shape != (self.state_dim,):
                # Pad or truncate to make sure we have the right size
                if len(state) < self.state_dim:
                    state = np.pad(state, (0, self.state_dim - len(state)))
                else:
                    state = state[:self.state_dim]
                    
        except Exception as e:
            print(f"Error in get_state: {e}")
            # Return a zero state on error
            state = np.zeros(self.state_dim)
            
        return state

    def compute_reward(self, collision, action):
        try:
            # Balanced reward weights with more emphasis on exploration
            SAFETY_COLLISION_PENALTY = -10.0  # Reduced from -50 to -10
            SAFETY_REWARD = 0.5  # Reduced baseline safety reward
            EFFICIENCY_SPEED_WEIGHT = 1.0  # Reduced from 2.0 to 1.0
            EFFICIENCY_LANE_CHANGE_BONUS = 0.5  # Reduced from 1.0 to 0.5
            TRAFFIC_FLOW_REWARD = 0.4  # Reduced from 0.8 to 0.4
            JERK_PENALTY_WEIGHT = 0.1  # Reduced from 0.2 to 0.1
            
            # Function to cap reward components to prevent extreme values
            def cap_reward(value, min_val=-5.0, max_val=5.0):
                return max(min_val, min(max_val, value))
            
            # Safety reward - less dominant
            if collision:
                R_safe = SAFETY_COLLISION_PENALTY
            else:
                R_safe = SAFETY_REWARD
            
            # Speed efficiency - stronger reward for matching target speed
            if self.target_speed > 0.1:  # Avoid division by very small numbers
                speed_ratio = self.speed / self.target_speed
            else:
                speed_ratio = 0.0
                
            # Cap speed ratio to avoid extreme values
            speed_ratio = cap_reward(speed_ratio, 0.0, 2.0)
            R_speed = 1.0 * speed_ratio - 0.5  # Less extreme range (-0.5 to +1.5)
            
            # Lane change reward - properly detect lane changes using stored previous lane
            R_lane_change = 0
            if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
                # Check if a lane change was attempted
                if self.lane_change_attempted:
                    # Check if lane actually changed by comparing current with previous
                    if self.prev_sublane != self.curr_sublane:
                        R_lane_change = EFFICIENCY_LANE_CHANGE_BONUS  # Reward for successful change
                        # Extra reward for strategic lane changes to faster lanes
                        try:
                            # Check if this lane is moving faster than previous
                            if self.curr_lane and self.prev_sublane != -1:
                                prev_lane_id = f"{self.curr_lane.split('_')[0]}_{self.prev_sublane}"
                                if prev_lane_id in self.lane_ids:
                                    prev_lane_speed = traci.lane.getLastStepMeanSpeed(prev_lane_id)
                                    curr_lane_speed = traci.lane.getLastStepMeanSpeed(self.curr_lane)
                                    
                                    # Extra reward if moved to a faster lane (reduced and capped)
                                    if curr_lane_speed > prev_lane_speed:
                                        R_lane_change += 0.2  # Reduced from 0.5
                        except:
                            pass
                    else:
                        # Attempted change but failed
                        R_lane_change = -0.05  # Reduced penalty for failed attempt
            
            # Traffic flow reward - new component encouraging keeping pace with surrounding traffic
            try:
                # Collect speeds by lane and direction
                ahead_speeds = []
                behind_speeds = []  # Explicitly track vehicles behind
                same_lane_speeds = []
                other_lane_speeds = []
                
                # Get the agent's current lane info
                agent_lane = self.curr_sublane
                agent_edge = self.curr_lane.split("_")[0] if "_" in self.curr_lane else "e"
                
                for veh in traci.vehicle.getIDList():
                    if veh == self.name:
                        continue
                        
                    try:
                        # Get vehicle position and lane
                        veh_pos = traci.vehicle.getPosition(veh)
                        veh_lane = traci.vehicle.getLaneID(veh)
                        veh_sublane = int(veh_lane.split("_")[1]) if "_" in veh_lane else 0
                        veh_edge = veh_lane.split("_")[0] if "_" in veh_lane else "e"
                        
                        # Only consider vehicles on the same edge
                        if veh_edge != agent_edge:
                            continue
                            
                        # Calculate distance and get vehicle speed
                        dist = get_distance(self.pos, veh_pos)
                        veh_speed = traci.vehicle.getSpeed(veh)
                        
                        # Skip vehicles too far away
                        if dist > 100:
                            continue
                            
                        # Determine if vehicle is ahead or behind
                        # In SUMO, angle 0 means East, so we need to check relative positions
                        veh_angle = angle_between(self.pos, veh_pos, self.angle)
                        is_ahead = (veh_angle > 337.5 or veh_angle < 22.5)
                        is_behind = (veh_angle > 157.5 and veh_angle < 202.5)
                        
                        # Explicitly track vehicles behind
                        if is_behind and dist < 50:
                            behind_speeds.append((veh_speed, dist))
                        
                        # Categorize the vehicle
                        if veh_sublane == agent_lane:
                            # Same lane
                            same_lane_speeds.append((veh_speed, dist, is_ahead))
                        else:
                            # Different lane
                            other_lane_speeds.append((veh_speed, dist, is_ahead))
                            
                        # Keep track of vehicles ahead (regardless of lane)
                        if is_ahead and dist < 50:
                            ahead_speeds.append((veh_speed, dist))
                            
                    except:
                        continue
                
                # Calculate a weighted traffic flow reward
                R_traffic = 0
                
                # Component 1: Reward for keeping pace with same-lane traffic
                if same_lane_speeds:
                    # Weight by distance (closer vehicles matter more)
                    total_weight = 0
                    weighted_speed = 0
                    
                    for speed, dist, is_ahead in same_lane_speeds:
                        # Higher weight for vehicles ahead
                        weight = (1.0 / max(dist, 5)) * (2.0 if is_ahead else 1.0)
                        weighted_speed += speed * weight
                        total_weight += weight
                        
                    if total_weight > 0:
                        avg_same_lane_speed = weighted_speed / total_weight
                        if self.target_speed > 0.1:  # Avoid division by very small numbers
                            same_lane_reward = -abs(self.speed - avg_same_lane_speed) / self.target_speed
                            same_lane_reward = cap_reward(same_lane_reward, -1.0, 0.0)
                            R_traffic += 0.6 * same_lane_reward  # 60% weight for same-lane
                
                # Component 2: Reward for considering other lanes
                if other_lane_speeds:
                    total_weight = 0
                    weighted_speed = 0
                    
                    for speed, dist, is_ahead in other_lane_speeds:
                        # Only moderate weight for other lanes
                        weight = (1.0 / max(dist, 10)) * (1.5 if is_ahead else 0.5)
                        weighted_speed += speed * weight
                        total_weight += weight
                        
                    if total_weight > 0:
                        avg_other_lane_speed = weighted_speed / total_weight
                        if self.target_speed > 0.1:  # Avoid division by very small numbers
                            other_lane_reward = -abs(self.speed - avg_other_lane_speed) / self.target_speed
                            other_lane_reward = cap_reward(other_lane_reward, -1.0, 0.0)
                            R_traffic += 0.3 * other_lane_reward  # 30% weight for other lanes
                
                # Component 3: Penalty for being significantly slower than vehicles ahead
                if ahead_speeds:
                    min_ahead_dist = min([dist for _, dist in ahead_speeds])
                    avg_ahead_speed = np.mean([spd for spd, _ in ahead_speeds])
                    
                    # If we're creating a bottleneck (much slower than traffic ahead)
                    if self.speed < avg_ahead_speed * 0.7 and min_ahead_dist > 20:
                        # Significantly reduced penalty for blocking traffic
                        if self.target_speed > 0.1:
                            bottleneck_penalty = 0.1 * (avg_ahead_speed - self.speed) / self.target_speed
                            bottleneck_penalty = cap_reward(bottleneck_penalty, 0.0, 0.5)
                            R_traffic -= bottleneck_penalty
                
                # NEW - Component 4: Special traffic jam detection for stopped agent (REDUCED PENALTY)
                if self.speed < 0.5:  # Agent is basically stopped
                    if behind_speeds:
                        # Count stopped vehicles behind
                        stopped_count = sum(1 for spd, _ in behind_speeds if spd < 0.5)
                        total_behind = len(behind_speeds)
                        
                        # If there are stopped vehicles behind us (traffic jam)
                        if stopped_count > 0:
                            # Calculate what percentage of vehicles behind are stopped
                            jam_ratio = stopped_count / total_behind
                            # Heavily reduced penalty for jam - capped to prevent extreme values
                            jam_penalty = 0.2 * jam_ratio * min(1.0, stopped_count / 10.0)
                            jam_penalty = cap_reward(jam_penalty, 0.0, 0.5)
                            R_traffic -= jam_penalty
                            print(f"Traffic jam detected: {stopped_count}/{total_behind} vehicles stopped behind agent. Penalty: {jam_penalty:.2f}")
                
                # If low speed and vehicles are present but no traffic measurement was made
                if R_traffic == 0 and (ahead_speeds or behind_speeds or same_lane_speeds):
                    if self.speed < 0.5:  # Very slow or stopped
                        # Apply reduced penalty for stopping in traffic
                        nearby_count = len(ahead_speeds) + len(behind_speeds) + len(same_lane_speeds)
                        stopped_penalty = -0.1 * min(nearby_count / 10.0, 0.5)
                        R_traffic = cap_reward(stopped_penalty, -0.5, 0.0)
                    elif self.speed > 0 and self.target_speed > 0.1:
                        # Default to a small positive reward for movement when in traffic
                        R_traffic = min(0.1 * (self.speed / self.target_speed), 0.2)
                
                # Final cap on traffic reward
                R_traffic = cap_reward(R_traffic, -1.0, 0.5)
                
            except Exception as e:
                print(f"Error calculating traffic flow: {e}")
                R_traffic = 0
            
            # Jerk penalty (reduced weight)
            jerk = self.compute_jerk()
            R_jerk = -np.abs(jerk) / 5.0  # Further reduced penalty
            R_jerk = cap_reward(R_jerk, -0.5, 0.0)  # Cap jerk penalty
            
            # Normalize rewards to [-1, 1] range
            def normalize_reward(value, min_val, max_val):
                return 2 * (value - min_val) / (max_val - min_val) - 1
            
            R_total = (
                normalize_reward(R_safe, -10.0, 1.0) * 0.3 +
                normalize_reward(R_speed, -2.0, 2.0) * 0.3 +
                normalize_reward(R_lane_change, -0.5, 1.0) * 0.2 +
                normalize_reward(R_traffic, -2.0, 1.0) * 0.15 +
                normalize_reward(R_jerk, -0.5, 0.0) * 0.05
            )
            
            # Final cap on total reward to ensure stability
            R_total = cap_reward(R_total, -10.0, 5.0)
            
            # Add debugging for extreme values
            if abs(R_total) > 5.0 or abs(R_safe) > 5.0 or abs(R_speed) > 2.0 or abs(R_traffic) > 2.0:
                print(f"WARNING: Large reward components - Total:{R_total:.2f}, Safe:{R_safe:.2f}, Speed:{R_speed:.2f}, Traffic:{R_traffic:.2f}")
            
            return R_total, R_jerk, R_speed, R_safe, R_traffic
            
        except Exception as e:
            print(f"Error in compute_reward: {e}")
            return -1, 0, 0, -1, 0