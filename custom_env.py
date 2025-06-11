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
from pettingzoo import ParallelEnv
import random

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

class MultiAgentCarEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "sumo_traffic_v0"}
    
    def __init__(self, num_agents=3):
        super().__init__()
        self.possible_agents = [f'rlagent_{i}' for i in range(num_agents)]
        self.agents = self.possible_agents.copy()
        
        self.step_length = 0.4
        self.acc_history = {agent: deque([0, 0], maxlen=2) for agent in self.possible_agents}
        self.grid_state_dim = 3
        self.grid_cells = self.grid_state_dim * self.grid_state_dim  # 9 cells
        self.features_per_vehicle = 4  # [dist/pos_x, pos_y/speed, speed/acc, lat_speed/lat_pos]
        self.agent_features = 5  # [pos_x, pos_y, speed, lat_speed, acc]
        self.state_dim = (self.grid_cells * self.features_per_vehicle) + 1
        
        # Validate state dimension
        assert self.state_dim == 37, f"Expected state dimension 37, got {self.state_dim}"
        
        # Initialize per-agent parameters
        self.pos = {agent: (0, 0) for agent in self.possible_agents}
        self.curr_lane = {agent: '' for agent in self.possible_agents}
        self.curr_sublane = {agent: -1 for agent in self.possible_agents}
        self.prev_sublane = {agent: -1 for agent in self.possible_agents}
        self.target_speed = {agent: 0 for agent in self.possible_agents}
        self.speed = {agent: 0 for agent in self.possible_agents}
        self.lat_speed = {agent: 0 for agent in self.possible_agents}
        self.acc = {agent: 0 for agent in self.possible_agents}
        self.angle = {agent: 0 for agent in self.possible_agents}
        self.lane_change_attempted = {agent: False for agent in self.possible_agents}
        self.lane_change_cooldown = {agent: 0 for agent in self.possible_agents}
        self.collision = {agent: False for agent in self.possible_agents}
        
        self.gui = True
        self.numVehicles = 0
        self.vType = 0
        self.lane_ids = []
        self.max_steps = 10000
        self.curr_step = 0
        self.done = False
        self.traci_started = False
        self.sumo_process = None
        self.lane_change_cooldown_steps = 10  # Minimum steps between lane changes (4 seconds at 0.4s step)
        
        # Register an exit handler to ensure SUMO closes on program exit
        atexit.register(self.close_on_exit)

        # Define observation spaces for each agent
        self.observation_spaces = {
            agent: spaces.Box(
            low=-np.inf, 
            high=np.inf, 
                shape=(self.state_dim,),
            dtype=np.float32
            ) for agent in self.possible_agents
        }

        # Define action spaces for each agent
        self.action_spaces = {
            agent: spaces.Box(
            low=-1,
            high=1,
                shape=(2,),  # [acceleration, lane_change]
            dtype=np.float32
            ) for agent in self.possible_agents
        }
        
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
            
        # Fix paths - convert to absolute paths which SUMO requires
        absolute_network_conf = os.path.abspath(network_conf)
        absolute_network_xml = os.path.abspath(network_xml)
        
        print(f"Using SUMO config: {absolute_network_conf}")
        print(f"Using network file: {absolute_network_xml}")
            
        self.net = sumolib.net.readNet(absolute_network_xml)
        self.curr_step = 0
        self.done = False
    
        # Starting SUMO with appropriate parameters
        if gui:
            sumoCmd = ["sumo-gui", 
                      "-c", absolute_network_conf, 
                      "--no-warnings",  # Reduce noise
                      "--no-step-log", "true", 
                      "--start",  # Start immediately
                      "--quit-on-end",  # Allow SUMO to quit when closed
                      "--window-size", "1200,800",  # Set a reasonable window size
                      "--delay", "50",  # Add a small visualization delay so you can see what's happening
                      "--collision.action", "warn",  # Important: don't remove vehicles on collision
                      "--collision.mingap-factor", "0",  # Allow collisions to be detected
                      "--time-to-teleport", "-1",  # Disable teleporting
                      "--step-length", str(self.step_length)]
        else:
            # Use headless SUMO for faster training
            sumoCmd = ["sumo", 
                      "-c", absolute_network_conf, 
                      "--no-warnings",
                      "--no-step-log", "true",
                      "--quit-on-end",
                      "--collision.action", "warn",
                      "--collision.mingap-factor", "0",
                      "--time-to-teleport", "-1",  # Disable teleporting
                      "--step-length", str(self.step_length)]
        
        try:
            print("Starting SUMO...")
            # Start SUMO process using subprocess to better manage lifecycle
            if self.gui:
                # For GUI mode, let TraCI handle the connection
                traci.start(sumoCmd)
                
                # Give SUMO GUI time to initialize
                time.sleep(1.0)
                
                # Set SUMO window focus
                try:
                    # Try to bring the SUMO window to the foreground if possible
                    if hasattr(traci, 'gui'):
                        try:
                            # Set viewing area to show entire network
                            view_ids = traci.gui.getIDList()
                            if view_ids:
                                traci.gui.setView(view_ids[0], 0, 0, 100)  # Center view, zoom level 100
                                traci.gui.setFocus(view_ids[0])  # Focus the window
                        except Exception as e:
                            print(f"Warning: Could not set SUMO GUI view: {e}")
                except Exception as e:
                    print(f"Error setting GUI focus: {e}")  # Properly handle exception
            else:
                # For headless mode, use subprocess for better process management
                self.sumo_process = subprocess.Popen(sumoCmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Connect to SUMO using TraCI on a free port
                port = traci.getFreeSocketPort()
                traci.init(port=port)
                
            self.traci_started = True
            print("SUMO started successfully")
            
            # Get lane IDs if possible, with error handling
            try:
                self.lane_ids = traci.lane.getIDList()
                print(f"Found {len(self.lane_ids)} lanes in the network")
            except traci.exceptions.FatalTraCIError as e:
                print(f"TraCI connection failed: {e}")
                return False
            
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            return False
    
        # Run initial simulation steps to load the network
        print("Initializing simulation...")
        for _ in range(10):
            traci.simulationStep()
        
        # Clear any existing agents that might be in the simulation
        for agent_id in self.possible_agents:
            if agent_id in traci.vehicle.getIDList():
                try:
                    traci.vehicle.remove(agent_id)
                    print(f"Removed existing agent {agent_id} from simulation")
                except:
                    pass
        
        # Allow simulation to process removals
        traci.simulationStep()
            
        # Add RL agents with proper spacing and error handling
        self._add_rl_agents()
        
        # Run a few simulation steps to stabilize RL agent placement
        for _ in range(5):
            traci.simulationStep()
        
        # Verify all agents were added successfully
        added_agents = [agent for agent in self.possible_agents if agent in traci.vehicle.getIDList()]
        if len(added_agents) < len(self.possible_agents):
            print(f"Warning: Only {len(added_agents)}/{len(self.possible_agents)} RL agents were added successfully")
            print(f"Successfully added agents: {added_agents}")
        
        # Add regular vehicles with proper spacing
        self._add_regular_vehicles()
    
        # Distribute vehicles with additional simulation steps
        print("Distributing vehicles...")
        steps_completed = 0
        for step in range(30):  # Reduced from 50 to save time
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
        
        # Final verification of RL agents
        final_agents = [agent for agent in self.possible_agents if agent in traci.vehicle.getIDList()]
        print(f"Final check: {len(final_agents)}/{len(self.possible_agents)} RL agents in simulation")
    
        # Setting up useful parameters
        try:
            self.update_params()
            return True
        except Exception as e:
            print(f"Error updating parameters: {e}")
            return False
            
    def _add_rl_agents(self):
        print(f"Adding {len(self.possible_agents)} RL agents to simulation...")
        
        for i, agent in enumerate(self.possible_agents):
            try:
                # Use different departure positions with much more spacing
                # Place RL agents far apart (100m between them)
                departPos = 50 + (i * 100)  # Starting at 50m position, 100m between agents
                
                # Distribute across available lanes - ensure each agent has a different lane
                departLane = i % min(3, len(self.lane_ids))
                
                # Safety check - is this lane available?
                if len([lane for lane in self.lane_ids if f"_{departLane}" in lane]) == 0:
                    departLane = 0  # Default to first lane
                
                # Double-check agent doesn't exist before adding
                if agent in traci.vehicle.getIDList():
                    try:
                        traci.vehicle.remove(agent)
                        print(f"Removed duplicate agent {agent} before adding")
                        traci.simulationStep()  # Process removal
                    except:
                        pass
                
                print(f"Adding agent {agent} at position {departPos}m on lane {departLane}")
                traci.vehicle.add(agent, routeID='route_0', typeID='rl', 
                                  departLane=str(departLane), 
                                  departPos=str(departPos),
                                  departSpeed="0")  # Start with zero speed to avoid departure errors
                
                # Setting the lane change mode for the RL agent (512 = only strategic changes by vehicle)
                try:
                    if agent in traci.vehicle.getIDList():
                        traci.vehicle.setLaneChangeMode(agent, 512)
                except traci.exceptions.TraCIException:
                    # Vehicle might have disappeared or moved during initialization
                    # This is expected occasionally - just continue
                    pass
                
                # Step simulation to ensure agent insertion
                traci.simulationStep()
                
                # Now set speed after insertion rather than at departure
                if agent in traci.vehicle.getIDList():
                    initial_speed = max(3, 5 + i * 2)  # Start slower, vary initial speeds
                    traci.vehicle.setSpeed(agent, initial_speed)
                    # Add a minimum gap to other vehicles
                    traci.vehicle.setMinGap(agent, 5.0)  # 5 meter minimum gap
                else:
                    print(f"Warning: Agent {agent} not found after insertion step")
                    # Try once more with different departure parameters
                    try:
                        departPos = 50 + (i * 100) + 20  # Add offset to avoid conflicts
                        traci.vehicle.add(agent, routeID='route_0', typeID='rl', 
                                          departLane=str(departLane), 
                                          departPos=str(departPos),
                                          departSpeed="0")
                        traci.vehicle.setLaneChangeMode(agent, 512)
                        traci.simulationStep()
                    except Exception as retry_error:
                        print(f"Retry to add agent {agent} failed: {retry_error}")
                
            except traci.exceptions.TraCIException as e:
                print(f"Error adding RL agent {agent}: {e}")
                # Try one more time with different parameters
                try:
                    # Try different position and lane
                    alt_departPos = 100 + (i * 120)
                    alt_departLane = (i + 1) % min(3, len(self.lane_ids))
                    
                    print(f"Retrying agent {agent} at alt position {alt_departPos}m on lane {alt_departLane}")
                    traci.vehicle.add(agent, routeID='route_0', typeID='rl', 
                                      departLane=str(alt_departLane), 
                                      departPos=str(alt_departPos),
                                      departSpeed="5")
                    
                    traci.vehicle.setLaneChangeMode(agent, 512)
                    traci.simulationStep()
                except:
                    print(f"Failed to add agent {agent} after multiple attempts")
    
    def _add_regular_vehicles(self):
        """Helper method to add regular vehicles to the simulation"""
        # Add regular traffic with lower density
        # Only add 2/3 of the requested number to reduce density
        actual_vehicles = max(5, int(self.numVehicles * 0.6))  # At least 5, but 60% of requested
        print(f"Adding {actual_vehicles} regular vehicles (requested {self.numVehicles})")
        
        vehicles_added = 0
        
        # Place vehicles with more spacing
        for i in range(actual_vehicles):
            veh_name = 'vehicle_' + str(i)
            
            # Check if vehicle already exists
            if veh_name in traci.vehicle.getIDList():
                try:
                    traci.vehicle.remove(veh_name)
                except:
                    pass
                
            # Distribute over a much wider range of positions
            pos = 10 + (i * 30)  # 30 meters between vehicles
            lane = i % min(3, len(self.lane_ids))
            
            try:
                # Use explicit departure position to ensure spacing
                traci.vehicle.add(veh_name, routeID='route_0', typeID=self.vType, 
                                 departLane=str(lane), 
                                 departPos=str(pos),
                                 departSpeed="random")
                
                # Make vehicles continue driving by setting "keepRoute" mode
                traci.vehicle.setLaneChangeMode(veh_name, 256)
                vehicles_added += 1
                
                # Add minimum gap to other vehicles
                traci.vehicle.setMinGap(veh_name, 2.5)
                
                # Step simulation every few vehicles to distribute them
                if i % 5 == 0:
                    traci.simulationStep()
                    
            except traci.exceptions.TraCIException as e:
                print(f"Error adding vehicle {veh_name}: {e}")
                
        if vehicles_added == 0:
            print("Failed to add any regular vehicles. Check your route definitions.")
            # Continue anyway since we have RL agents
    
    def is_sumo_running(self):
        """Check if SUMO is still running"""
        try:
            # Check if we can still get simulation time - will raise exception if connection is gone
            time_value = traci.simulation.getTime()
            
            # Check if traci connection is still valid by checking if we can get vehicle IDs
            veh_list = traci.vehicle.getIDList()
            
            # Both checks passed - SUMO is running
            self.traci_started = True
            return True
            
        except traci.exceptions.FatalTraCIError:
            # Connection is definitely dead
            self.traci_started = False
            return False
        except Exception as e:
            # Any other error indicates connection issues
            print(f"Error checking if SUMO is running: {e}")
            self.traci_started = False
            return False
        
    def update_params(self):
        # initialize params
        try:
            # First check if SUMO is still running
            if not self.is_sumo_running():
                self.traci_started = False
                return
                
            # Now try to update parameters for each agent
            for agent in self.possible_agents:
                if agent not in traci.vehicle.getIDList():
                    # If the RL agent is not in the simulation, try to add it back
                    try:
                        traci.vehicle.add(agent, routeID='route_0', typeID='rl')
                        traci.vehicle.setLaneChangeMode(agent, 0)
                        # Give SUMO a step to register the vehicle
                        traci.simulationStep()
                    except:
                        pass
                
                # Get position and lane safely
                try:
                    self.pos[agent] = traci.vehicle.getPosition(agent)
                    self.curr_lane[agent] = traci.vehicle.getLaneID(agent)
                except:
                    # Default values if vehicle doesn't exist
                    self.pos[agent] = (0, 0)
                    self.curr_lane[agent] = "e_0"  # Default lane
                
                if not self.curr_lane[agent]:
                    # Handle case when the vehicle is in teleport state
                    if agent in traci.simulation.getStartingTeleportIDList():
                        # Wait for teleport to finish
                        teleport_waiting_steps = 0
                        while agent in traci.simulation.getStartingTeleportIDList() and teleport_waiting_steps < 10:
                            traci.simulationStep()
                            teleport_waiting_steps += 1
                        try:
                            self.curr_lane[agent] = traci.vehicle.getLaneID(agent)
                        except:
                            self.curr_lane[agent] = "e_0"  # Default if still issues
                    else:
                        # Vehicle might be removed, try to add it back
                        try:
                            traci.vehicle.add(agent, routeID='route_0', typeID='rl')
                            traci.vehicle.setLaneChangeMode(agent, 0)
                            traci.simulationStep()
                            self.curr_lane[agent] = traci.vehicle.getLaneID(agent)
                        except:
                            # If we can't add it back, use default values
                            self.curr_lane[agent] = "e_0"  # Default lane
                
                # Parse sublane index from lane ID
                try:
                    if '_' in self.curr_lane[agent]:
                        # Extract lane index and ensure it's non-negative
                        lane_index = self.curr_lane[agent].split("_")[1]
                        # Make sure it's a valid integer and positive
                        try:
                            self.curr_sublane[agent] = max(0, int(lane_index))
                        except ValueError:
                            self.curr_sublane[agent] = 0
                    else:
                        self.curr_sublane[agent] = 0  # Default value
                except (IndexError, ValueError):
                    self.curr_sublane[agent] = 0  # Default value
                
                # Get other vehicle parameters safely with defaults
                try:
                    self.target_speed[agent] = traci.vehicle.getAllowedSpeed(agent)
                except:
                    self.target_speed[agent] = 13.89  # Default 50km/h in m/s
                    
                try:
                    self.speed[agent] = traci.vehicle.getSpeed(agent)
                except:
                    self.speed[agent] = 0
                    
                try:
                    self.lat_speed[agent] = traci.vehicle.getLateralSpeed(agent)
                except:
                    self.lat_speed[agent] = 0
                    
                try:
                    self.acc[agent] = traci.vehicle.getAcceleration(agent)
                except:
                    self.acc[agent] = 0
                    
                self.acc_history[agent].append(self.acc[agent])
                
                try:
                    self.angle[agent] = traci.vehicle.getAngle(agent)
                except:
                    self.angle[agent] = 0
                
        except traci.exceptions.TraCIException as e:
            print(f"Error in update_params: {e}")
            # Use default values if there's an error
            for agent in self.possible_agents:
                if len(self.acc_history[agent]) < 2:
                    self.acc_history[agent].append(0)
                if not hasattr(self, 'pos') or not self.pos[agent]:
                    self.pos[agent] = (0, 0)

    def get_grid_state(self, agent_name, threshold_distance=10):
        '''
        Observation is a grid occupancy grid for the specified agent
        '''
        agent_lane = self.curr_lane[agent_name]
        agent_pos = self.pos[agent_name]
        
        # Default to edge "e" if curr_lane is invalid
        if not self.curr_lane[agent_name] or not "_" in self.curr_lane[agent_name]:
            edge = "e"
        else:
            edge = self.curr_lane[agent_name].split("_")[0]
            
        agent_lane_index = self.curr_sublane[agent_name]
        
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
        
        # Put other vehicles (including other RL agents)
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
                if vehicle == agent_name:
                    continue
                    
                try:
                    # Get angle wrt current agent
                    veh_pos = traci.vehicle.getPosition(vehicle)
                    # If too far, continue
                    if get_distance(agent_pos, veh_pos) > threshold_distance:
                        continue
                        
                    rl_angle = traci.vehicle.getAngle(agent_name)
                    
                    # Extract vehicle ID - handle both RL agents and normal vehicles
                    if "rlagent_" in vehicle:
                        # It's another RL agent - use -2 to distinguish from primary agent (-1)
                        veh_id = -2  
                    else:
                        # Regular vehicle - extract numeric ID if possible
                        try:
                            veh_id = int(vehicle.split("_")[1])
                        except (IndexError, ValueError):
                            veh_id = 1  # Default ID for unrecognized vehicles
                        
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
    
    def render(self):
        """
        Render the environment
        In this case, SUMO-GUI handles the rendering automatically
        """
        pass
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment according to PettingZoo API.
        Args:
            seed: The seed for randomization
            options: Additional options for environment reset (e.g., gui, numVehicles)
                    force_restart: Whether to force restart SUMO even if it's already running
        Returns:
            observations: Dict mapping agent_id to observation
            infos: Dict mapping agent_id to info dictionary
        """
        # Handle seeding
        if seed is not None:
            super().reset(seed=seed) # PettingZoo handles seeding internally
            np.random.seed(seed)
            # Note: Traci/SUMO itself might not be fully deterministic even with seed

        # Get options with defaults
        if options is None: options = {}
        gui = options.get('gui', True)
        numVehicles = options.get('numVehicles', np.random.randint(20, 35))
        vType = options.get('vType', 'human')
        # New option to force restart even if SUMO is already running
        force_restart = options.get('force_restart', False)
        
        if force_restart:
            print("Force restart requested - closing and restarting SUMO")
            # Always close regardless of current state when force_restart is true
            self.close()
            restart_sumo = True
        else:
            # Only restart if SUMO is not already running
            restart_sumo = not self.is_sumo_running()
            if not restart_sumo:
                print("Using existing SUMO simulation - resetting agent state only")
        
        if restart_sumo:
            print("Starting new SUMO simulation")
            # Wait a moment to ensure resources are released
            time.sleep(1.0)
            
            # Start SUMO simulation
            success = False
            for attempt in range(3):
                try:
                    success = self.start(gui=gui, numVehicles=numVehicles, vType=vType)
                    if success:
                        break
                except Exception as e:
                    print(f"Error during self.start() on attempt {attempt+1}: {e}")
                print(f"Attempt {attempt+1} failed. Retrying...")
                time.sleep(1)
                
            if not success:
                print("CRITICAL: Failed to start SUMO simulation after multiple attempts during reset.")
                # Return empty dicts or handle error appropriately
                return {}, {}
        else:
            # Just do a soft reset of the agent state without restarting SUMO
            # Reset internal counter
            self.curr_step = 0
            # Make sure agents state is reset
            self.agents = self.possible_agents.copy()
            # Reset collision state and other agent parameters
            self.collision = {agent: False for agent in self.possible_agents}
            # Don't reset position and lane info - we want to keep the current state
            # Reset the acceleration history
            self.acc_history = {agent: deque([0, 0], maxlen=2) for agent in self.possible_agents}
            # Rest lane change state
            self.lane_change_attempted = {agent: False for agent in self.possible_agents}
            self.lane_change_cooldown = {agent: 0 for agent in self.possible_agents}
            
            # Update params to make sure all variables are current
            self.update_params()

        # Reset internal state variables if this is a full restart
        if restart_sumo:
            self.agents = self.possible_agents.copy() # Crucial: Reset the list of active agents
            self.curr_step = 0
            self.collision = {agent: False for agent in self.possible_agents}
            self.acc_history = {agent: deque([0, 0], maxlen=2) for agent in self.possible_agents}
            self.pos = {agent: (0, 0) for agent in self.possible_agents}
            self.curr_lane = {agent: '' for agent in self.possible_agents}
            self.curr_sublane = {agent: -1 for agent in self.possible_agents}
            self.prev_sublane = {agent: -1 for agent in self.possible_agents}
            self.target_speed = {agent: 0 for agent in self.possible_agents}
            self.speed = {agent: 0 for agent in self.possible_agents}
            self.lat_speed = {agent: 0 for agent in self.possible_agents}
            self.acc = {agent: 0 for agent in self.possible_agents}
            self.angle = {agent: 0 for agent in self.possible_agents}
            self.lane_change_attempted = {agent: False for agent in self.possible_agents}
            self.lane_change_cooldown = {agent: 0 for agent in self.possible_agents}

        # Get initial observations and infos for all agents
        observations = {}
        infos = {}
        try:
            # Update params to get initial state from SUMO
            self.update_params() 
            
            active_agents_in_sumo = traci.vehicle.getIDList()
            initial_agents = []
            
            for agent_id in self.possible_agents:
                if agent_id in active_agents_in_sumo:
                    observations[agent_id] = self.get_state(agent_name=agent_id)
                    infos[agent_id] = {"initial_state": True} # Add any relevant initial info
                    initial_agents.append(agent_id)
                else:
                    # Agent couldn't be added or disappeared immediately
                    print(f"Warning: Agent {agent_id} not found in SUMO immediately after start.")
                    # Provide a default observation and info, but don't add to active agents list yet
                    observations[agent_id] = np.zeros(self.observation_spaces[agent_id].shape, dtype=np.float32)
                    infos[agent_id] = {"initial_state": False, "error": "not_found_at_start"}

            # Update the active agents list based on who is actually present
            self.agents = initial_agents
            if not self.agents:
                 print("Warning: No RL agents were successfully added or found at the start of the episode.")
                 return observations, infos # Return observations/infos even if agents list is empty initially

        except Exception as e:
            print(f"Error during initial state retrieval in reset: {e}")
            # If state retrieval fails, return empty observations/infos for active agents
            observations = {agent_id: np.zeros(self.observation_spaces[agent_id].shape, dtype=np.float32) for agent_id in self.agents}
            infos = {agent_id: {"error": "reset_state_retrieval_failed"} for agent_id in self.agents}
            self.agents = [] # Mark as no active agents if setup failed

        return observations, infos
    
    def close(self):
        """Clean up TraCI and terminate SUMO process"""
        print("Closing SUMO environment...")
        
        # Check if TraCI connection is active
        if self.traci_started:
            try:
                # Try to end simulation first if still running
                try:
                    if self.is_sumo_running():
                        traci.close()
                        print("TraCI connection closed successfully")
                except:
                    print("Error while closing TraCI connection gracefully")
                
                # Make sure connection is closed regardless
                if hasattr(traci, 'close'):
                    traci.close()
            except Exception as e:
                print(f"Error closing TraCI connection: {e}")
                
            self.traci_started = False
            
        # Kill SUMO process if it exists
        if self.sumo_process is not None:
            print("Terminating SUMO process...")
            try:
                # Try graceful termination first
                self.sumo_process.terminate()
                try:
                    # Wait up to 3 seconds for process to terminate
                    self.sumo_process.wait(timeout=3)
                    print("SUMO process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate in time
                    print("SUMO process didn't terminate in time, killing forcefully")
                    self.sumo_process.kill()
                    self.sumo_process.wait(timeout=1)
            except Exception as e:
                print(f"Error terminating SUMO process: {e}")
                try:
                    # Last resort: force kill
                    self.sumo_process.kill()
                except:
                    pass
                    
            self.sumo_process = None
            
        # Give system time to clean up ports, etc.
        time.sleep(1.0)
        print("SUMO environment closed")
        
        # Reset flags
        self.traci_started = False
    
    def compute_jerk(self, agent_name):
        """Compute jerk for an agent"""
        if len(self.acc_history[agent_name]) < 2:
            return 0
        return (self.acc_history[agent_name][1] - self.acc_history[agent_name][0])/self.step_length
    
    def detect_collision(self, agent_name):
        """Detect if the given agent has collided"""
        try:
            collisions = traci.simulation.getCollidingVehiclesIDList()
            if agent_name in collisions:
                # Get collision details if possible
                try:
                    collision_vehicles = []
                    for veh in collisions:
                        if veh != agent_name and veh in traci.vehicle.getIDList():
                            # Get the colliding vehicle's info
                            veh_speed = traci.vehicle.getSpeed(veh)
                            veh_lane = traci.vehicle.getLaneID(veh)
                            veh_pos = traci.vehicle.getPosition(veh)
                            dist = get_distance(self.pos[agent_name], veh_pos)
                            collision_vehicles.append(f"{veh} (speed={veh_speed:.1f}, lane={veh_lane}, dist={dist:.1f})")
                    
                    # Log detailed collision info
                    print(f"COLLISION DETECTED: Agent {agent_name} speed={self.speed[agent_name]:.1f}, lane={self.curr_lane[agent_name]}")
                    print(f"Collided with: {', '.join(collision_vehicles) if collision_vehicles else 'Unknown'}")
                except:
                    print(f"COLLISION DETECTED: Agent {agent_name} - Could not get detailed collision info")
                
                self.collision[agent_name] = True
                return True
        except:
            # If we can't get collision info, assume no collision
            pass
            
        self.collision[agent_name] = False
        return False

    def step(self, actions):
        # actions: dictionary mapping agent_id to action array
        
        # Store current lane to detect changes later
        self.prev_sublane = {agent: self.curr_sublane.get(agent, -1) for agent in self.agents}
        # Reset lane change tracking
        self.lane_change_attempted = {agent: False for agent in self.agents}
        
        # Update lane change cooldown counter
        self.lane_change_cooldown = {agent: max(0, self.lane_change_cooldown.get(agent, 0) - 1) for agent in self.agents}
        
        # Check if SUMO is running
        if not self.is_sumo_running():
            print("SUMO closed unexpectedly. Ending episode.")
            # Return state for all agents, mark as terminated
            zero_state = np.zeros(self.state_dim)
            obs = {agent: zero_state for agent in self.agents}
            rewards = {agent: -10 for agent in self.agents}
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: True for agent in self.agents}
            infos = {agent: {"error": "SUMO disconnected", "done": True} for agent in self.agents}
            # Update self.agents to reflect termination
            self.agents = []
            return obs, rewards, terminations, truncations, infos
        
        # Apply actions for each agent
        for agent_id, action in actions.items():
            if agent_id not in self.agents or agent_id not in traci.vehicle.getIDList():
                continue # Skip agents that are no longer active or not in simulation
            
            # --- Process Acceleration ---
            try:
                if isinstance(action, (list, np.ndarray)) and len(action) >= 1:
                    acceleration = action[0] * 4.0
                    current_speed = traci.vehicle.getSpeed(agent_id)
                    target_speed = max(0, current_speed + acceleration * self.step_length)
                    traci.vehicle.setSpeed(agent_id, target_speed)
            except traci.exceptions.TraCIException as e:
                pass
            
            # --- Process Lane Change ---
            try:
                if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
                    # Only attempt lane change if cooldown is 0
                    if self.lane_change_cooldown[agent_id] == 0:
                        lane_change = action[1]
                        if abs(lane_change) > 0.5:  # Threshold for lane change
                            self.lane_change_attempted[agent_id] = True
                            
                            # Get current lane information
                            current_lane_id = traci.vehicle.getLaneID(agent_id)
                            if current_lane_id and '_' in current_lane_id:
                                try:
                                    # Parse lane info
                                    edge = current_lane_id.split("_")[0]
                                    current_lane_idx = int(current_lane_id.split("_")[1])
                                    
                                    # Find all lanes on this edge
                                    lanes_on_edge = [lane for lane in self.lane_ids if lane.startswith(f"{edge}_")]
                                    num_lanes = len(lanes_on_edge)
                                    
                                    # Calculate target lane - ensure it's valid
                                    target_lane_idx = current_lane_idx
                                    if lane_change < -0.5 and current_lane_idx > 0:
                                        # Change left
                                        target_lane_idx = current_lane_idx - 1
                                    elif lane_change > 0.5 and current_lane_idx < num_lanes - 1:
                                        # Change right
                                        target_lane_idx = current_lane_idx + 1
                                    
                                    # Only send change command if target lane is different and valid
                                    if target_lane_idx != current_lane_idx and 0 <= target_lane_idx < num_lanes:
                                        # Use changeLane with proper lane index, not relative direction
                                        traci.vehicle.changeLane(agent_id, target_lane_idx, 1.0)
                                        self.lane_change_cooldown[agent_id] = self.lane_change_cooldown_steps
                                except (ValueError, IndexError, traci.exceptions.TraCIException) as e:
                                    # Silently ignore lane change errors - they're expected occasionally
                                    pass
            except traci.exceptions.TraCIException:
                # Silently handle exceptions
                pass
        
        # Route recycling logic (keep this general for all vehicles)
        try:
            veh_list = traci.vehicle.getIDList()
            for veh in veh_list:
                try:
                    route_edges = traci.vehicle.getRoute(veh)
                    if not route_edges: continue
                    try: rem_distance = traci.vehicle.getRemainingDistance(veh)
                    except:
                        try:
                            route_idx = traci.vehicle.getRouteIndex(veh)
                            if route_idx >= len(route_edges) - 1: traci.vehicle.setRoute(veh, route_edges)
                        except: pass
                        continue
                    if rem_distance is not None and rem_distance < 50: traci.vehicle.setRoute(veh, route_edges)
                except: continue
        except traci.exceptions.TraCIException as e:
            pass
        
        # Advance simulation step
        try:
            traci.simulationStep()
            self.curr_step += 1
        except traci.exceptions.FatalTraCIError as e:
            print(f"Simulation step failed: {e}. Ending episode.")
            # Return state for all agents, mark as terminated
            zero_state = np.zeros(self.state_dim)
            obs = {agent: zero_state for agent in self.agents}
            rewards = {agent: -10 for agent in self.agents}
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: True for agent in self.agents}
            infos = {agent: {"error": "simulation_step_failed", "done": True} for agent in self.agents}
            self.agents = [] # Clear active agents
            return obs, rewards, terminations, truncations, infos
        
        # Update agent parameters and check status *after* simulation step
        try:
            self.update_params() # Updates internal state like position, speed, lane etc. for all possible agents
        except Exception as e:
            print(f"Error updating parameters after step: {e}")
        
        # --- Post-step processing for each agent ---
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        current_active_agents = traci.vehicle.getIDList() # Get list of vehicles currently in simulation
        
        agents_to_remove = []
        for agent_id in self.agents:
            agent_terminated = False
            agent_truncated = False
            agent_info = {}
            
            # Check if agent still exists in simulation
            if agent_id not in current_active_agents:
                # Try to re-add the agent after it disappeared
                try:
                    # Use last known position + a small offset, or default to a safe position
                    pos = self.pos.get(agent_id, (0, 0))
                    departPos = str(pos[0] + 10)  # Offset to avoid re-spawning at same location
                    
                    # Get a valid lane index
                    lane_idx = self.curr_sublane.get(agent_id, 0)
                    
                    # Ensure lane index is valid (check if the lane exists)
                    if lane_idx < 0:
                        lane_idx = 0  # Default to first lane if invalid
                    
                    # Find a valid edge for respawning
                    edge_name = "e"  # Default edge
                    if self.curr_lane.get(agent_id, ""):
                        if "_" in self.curr_lane[agent_id]:
                            edge_name = self.curr_lane[agent_id].split("_")[0]
                    
                    # Ensure the lane exists
                    test_lane_id = f"{edge_name}_{lane_idx}"
                    if test_lane_id not in self.lane_ids:
                        # Find a valid lane on this edge or default to lane 0
                        valid_lanes = [lane for lane in self.lane_ids if lane.startswith(f"{edge_name}_")]
                        if valid_lanes:
                            # Get the lane with the smallest valid index
                            lane_idx = min([int(lane.split("_")[1]) for lane in valid_lanes])
                        else:
                            lane_idx = 0  # Default if no valid lanes found
                        
                    departLane = str(lane_idx)
                    
                    # Try to add the vehicle back
                    traci.vehicle.add(agent_id, routeID='route_0', typeID='rl', 
                                    departLane=departLane, 
                                    departPos=departPos,
                                    departSpeed="10")
                    
                    # Set lane change mode and minimum gap
                    traci.vehicle.setLaneChangeMode(agent_id, 512)
                    traci.vehicle.setMinGap(agent_id, 5.0)
                    
                    # Give a small negative reward but don't terminate
                    rewards[agent_id] = -1.0
                    observations[agent_id] = np.zeros(self.observation_spaces[agent_id].shape, dtype=np.float32)
                    agent_info = {"status": "respawned"}
                    
                    # Successful re-add, run a simulation step to ensure it's in simulation
                    traci.simulationStep()
                    
                    # If successfully added, don't mark as terminated
                    if agent_id in traci.vehicle.getIDList():
                        agent_terminated = False
                        # Attempt to get a valid observation after respawn
                        try:
                            observations[agent_id] = self.get_state(agent_name=agent_id)
                        except:
                            pass
                        
                except traci.exceptions.TraCIException as e:
                    observations[agent_id] = np.zeros(self.observation_spaces[agent_id].shape, dtype=np.float32)
                    rewards[agent_id] = -2 # Penalty for disappearing but less than full termination
                    agent_info = {"status": "disappeared", "error": str(e)}
                    agent_terminated = True  # Only terminate if re-adding fails
            
            # If agent exists, get its state and compute reward
            if not agent_terminated:
                try:
                    observations[agent_id] = self.get_state(agent_name=agent_id)
                    
                    # Check for collisions
                    collision = False
                    try:
                        collision = traci.simulation.getCollidingVehiclesNumber() > 0
                    except:
                        pass
                    
                    # Store collision state
                    self.collision[agent_id] = collision
                    
                    # Compute reward
                    reward, r_jerk, r_eff, r_safe, r_traffic = self.compute_reward(collision, actions.get(agent_id, [0, 0]), agent_id)
                    rewards[agent_id] = reward
                    
                    # Add reward components to info
                    agent_info.update({
                        "r_jerk": r_jerk,
                        "r_eff": r_eff,
                        "r_safe": r_safe,
                        "r_traffic": r_traffic,
                        "collision": collision
                    })
                    
                except Exception as e:
                    print(f"Error computing state/reward for {agent_id}: {e}")
                    observations[agent_id] = np.zeros(self.observation_spaces[agent_id].shape, dtype=np.float32)
                    rewards[agent_id] = 0
                    agent_info = {"error": str(e)}
            
            # Add other standard info
            agent_info.update({
                "curr_lane": self.curr_lane.get(agent_id, "N/A"),
                "curr_sublane": self.curr_sublane.get(agent_id, -1),
                "speed": self.speed.get(agent_id, 0),
                "steps": self.curr_step,
                "lane_change_cooldown": self.lane_change_cooldown.get(agent_id, 0),
                "done": agent_terminated or agent_truncated or self.curr_step >= self.max_steps
            })
            
            # Check truncation
            if self.curr_step >= self.max_steps:
                agent_truncated = True
            
            terminations[agent_id] = agent_terminated
            truncations[agent_id] = agent_truncated
            infos[agent_id] = agent_info
            
            # Mark agent for removal if terminated or truncated
            if agent_terminated or agent_truncated:
                agents_to_remove.append(agent_id)
        
        # Update the list of active agents for the next step
        self.agents = [agent for agent in self.agents if agent not in agents_to_remove]
        
        return observations, rewards, terminations, truncations, infos

    def get_vehicle_info(self, vehicle_name, observer_agent=None):
        # Method to populate the vector information of a vehicle
        try:
            observer = observer_agent if observer_agent else self.possible_agents[0]
            
            if "rlagent_" in vehicle_name:
                # It's an RL agent
                try:
                    agent_idx = int(vehicle_name.split("_")[1])
                    agent = self.possible_agents[agent_idx]
                    return np.array([self.pos[agent][0], self.pos[agent][1], 
                                    self.speed[agent], self.lat_speed[agent], self.acc[agent]])
                except (IndexError, ValueError):
                    # Handle case where agent_idx is invalid or out of range
                    print(f"Warning: Invalid agent index for {vehicle_name}")
                    return np.zeros(5)
            else:
                # Check if vehicle exists
                if vehicle_name not in traci.vehicle.getIDList():
                    return np.zeros(4)  # Return zeros if vehicle doesn't exist
                    
                lat_pos, long_pos = traci.vehicle.getPosition(vehicle_name)
                long_speed = traci.vehicle.getSpeed(vehicle_name)
                acc = traci.vehicle.getAcceleration(vehicle_name)
                # Calculate distance from observer agent
                observer_pos = self.pos[observer]
                dist = get_distance(observer_pos, (lat_pos, long_pos))
                return np.array([dist, long_speed, acc, lat_pos])
        except Exception as e:
            # Return zeros on error
            if "rlagent_" in vehicle_name:
                return np.zeros(5)
            return np.zeros(4)
    
    def get_state(self, agent_name=None):
        # Define a state as a vector of vehicles information for the specified agent
        # If no agent specified, default to first agent for backward compatibility
        agent = agent_name if agent_name else self.possible_agents[0]
        
        state = np.zeros(self.state_dim)
        
        try:
            before = 0
            grid_state = self.get_grid_state(agent).flatten()
            
            for num, vehicle in enumerate(grid_state):
                if vehicle == 0:
                    continue
                if vehicle == -1:
                    vehicle_name = agent
                    before = 1
                elif vehicle == -2:
                    # Find which RL agent this is
                    nearest_agent = None
                    min_dist = float('inf')
                    agent_pos = self.pos[agent]
                    
                    for other_agent in self.possible_agents:
                        if other_agent == agent:
                            continue
                        other_pos = self.pos[other_agent]
                        dist = get_distance(agent_pos, other_pos)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_agent = other_agent
                    
                    if nearest_agent:
                        vehicle_name = nearest_agent
                    else:
                        continue  # Skip if we can't identify the agent
                else:
                    vehicle_name = f'vehicle_{int(vehicle)}'
                
                veh_info = self.get_vehicle_info(vehicle_name, agent)
                
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
            print(f"Error in get_state for {agent}: {e}")
            # Return a zero state on error
            state = np.zeros(self.state_dim)
            
        return state

    def compute_reward(self, collision, action, agent_name=None):
        agent = agent_name if agent_name else self.possible_agents[0]
        try:
            # Balanced reward weights with more emphasis on safety
            SAFETY_COLLISION_PENALTY = -3.0  # Reduced from -5.0 to keep training going
            SAFETY_REWARD = 0.5  # Increased to emphasize safe driving
            EFFICIENCY_SPEED_WEIGHT = 0.3  # Slightly reduced
            EFFICIENCY_LANE_CHANGE_BONUS = 0.2  # Reduced to discourage excessive lane changes 
            TRAFFIC_FLOW_REWARD = 0.2  
            JERK_PENALTY_WEIGHT = 0.05  
            
            # Function to cap reward components to prevent extreme values
            def cap_reward(value, min_val=-2.0, max_val=2.0):  # Less extreme caps
                return max(min_val, min(max_val, value))
            
            # Safety reward - higher emphasis
            if collision:
                R_safe = SAFETY_COLLISION_PENALTY
            else:
                R_safe = SAFETY_REWARD
            
            # Speed efficiency - smoother reward curve
            if self.target_speed[agent] > 0.1:  # Avoid division by very small numbers
                speed_ratio = self.speed[agent] / self.target_speed[agent]
            else:
                speed_ratio = 0.0
                
            # Cap speed ratio with smoother curve
            speed_ratio = cap_reward(speed_ratio, 0.0, 1.5)
            R_speed = 1.0 * speed_ratio - 0.3  # Smoother range (-0.3 to +1.2)
            
            # Lane change reward - properly detect lane changes using stored previous lane
            R_lane_change = 0
            if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
                # Only give lane change reward for successful changes
                if self.lane_change_attempted[agent] and self.prev_sublane[agent] != self.curr_sublane[agent]:
                    R_lane_change = EFFICIENCY_LANE_CHANGE_BONUS  # Small reward for successful change
            
            # Combine speed and lane change for efficiency
            R_eff = (R_speed * EFFICIENCY_SPEED_WEIGHT + R_lane_change)
            
            # Traffic flow reward component
            try:
                # Simpler traffic flow reward based on maintaining flow
                R_traffic = 0  # Default neutral value
                
                # If keeping reasonable pace with surrounding traffic
                try:
                    current_lane_id = self.curr_lane[agent]
                    if current_lane_id and current_lane_id in traci.lane.getIDList():
                        lane_mean_speed = traci.lane.getLastStepMeanSpeed(current_lane_id)
                        veh_count = traci.lane.getLastStepVehicleNumber(current_lane_id)
                        
                        # Only consider if there are other vehicles (avoid division by zero)
                        if veh_count > 1 and lane_mean_speed > 0:
                            # Reward for keeping pace with traffic
                            speed_diff = abs(self.speed[agent] - lane_mean_speed)
                            flow_reward = 0.2 * (1.0 - min(1.0, speed_diff / max(5.0, lane_mean_speed)))
                            R_traffic += flow_reward
                except Exception as e:
                    # If lane data unavailable, use a small positive reward for movement
                    if self.speed[agent] > 0:
                        R_traffic = 0.05
                
                # Final cap on traffic reward
                R_traffic = cap_reward(R_traffic, -0.5, 0.5)
                
            except Exception as e:
                R_traffic = 0
            
            # Jerk penalty (less weight)
            jerk = self.compute_jerk(agent)
            R_jerk = -np.abs(jerk) / 10.0  # Further reduced penalty
            R_jerk = cap_reward(R_jerk, -0.3, 0.0)  # Cap jerk penalty
            
            # Normalize rewards to [-1, 1] range with higher weight for safety
            def normalize_reward(value, min_val, max_val):
                value = np.clip(value, min_val, max_val)
                normalized = 2 * (value - min_val) / (max_val - min_val) - 1
                return np.clip(normalized, -1.0, 1.0)
            
            # Calculate final safety and efficiency indices
            safety_index = normalize_reward(R_safe, -3.0, 1.0)
            efficiency_index = normalize_reward(R_eff, -1.0, 1.0)
            
            # Weighted combination with higher weight on safety (0.4)
            R_total = (
                safety_index * 0.4 + 
                efficiency_index * 0.3 + 
                normalize_reward(R_traffic, -0.5, 0.5) * 0.2 + 
                normalize_reward(R_jerk, -0.3, 0.0) * 0.1
            )
            
            # Final cap on total reward to ensure stability - less extreme range
            R_total = np.clip(R_total, -2.0, 2.0)
            
            # Add debugging for extreme values
            if abs(R_total) > 2.0 or abs(safety_index) > 1.0 or abs(efficiency_index) > 1.0:
                print(f"WARNING: Large reward components for {agent} - Total:{R_total:.2f}, Safe:{safety_index:.2f}, Eff:{efficiency_index:.2f}")
            
            return R_total, R_jerk, efficiency_index, safety_index, R_traffic
            
        except Exception as e:
            print(f"Error in compute_reward for {agent}: {e}")
            return -1, 0, 0, -1, 0