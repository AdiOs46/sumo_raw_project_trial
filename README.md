*still a work in progress :P*
# Autonomous Driving Reinforcement Learning Project

This project implements a reinforcement learning model for autonomous driving using SUMO (Simulation of Urban MObility) traffic simulator and Stable Baselines3 - based on an existing repository (https://github.com/federicovergallo/SUMO-changing-lane-agent).

## Overview

The system aims to train an autonomous vehicle agent to navigate in traffic using Soft Actor-Critic (SAC) reinforcement learning. The agent aims to balance safety, comfort, efficiency, and traffic flow while driving in a simulated highway environment.

## Prerequisites

Before you begin, ensure you have the following installed:

### 1. Python Environment

- Python 3.8 or newer
- It's recommended to use a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

### 2. SUMO Traffic Simulator

SUMO is required to run the traffic simulations.

### 3. Python Dependencies

Install all required Python packages:

```bash
pip install gymnasium numpy matplotlib stable-baselines3 scipy sumolib traci
```

## Running the Project

### Training a Model

To train a new reinforcement learning model:

```bash
python train.py
```

This will start training with the following outputs:
- Trained models saved in the `models/` directory

### Evaluating a Trained Model

To evaluate a previously trained model:

```bash
python test.py
```

## Project Structure

- `custom_env.py`: Contains the custom Gymnasium environment for the autonomous driving agent
- `train.py`: Script to train the reinforcement learning model
- `test.py`: Script to evaluate trained models
- `models/`: Directory where trained models are saved

## Customization

You can modify various parameters in the code:
- In `train.py`: Adjust learning rates, neural network architecture, and training duration
- In `custom_env.py`: Modify reward functions, observation space, and simulation parameters

## Troubleshooting

- If you encounter SUMO connection errors, ensure SUMO is properly installed and SUMO_HOME is set correctly
- For visualization issues, check that the correct SUMO-GUI version is installed
- If you experience crashes during training, try lowering the number of vehicles or adjusting the step length
