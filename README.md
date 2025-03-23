# Elevator Simulator

## Overview

The Elevator Simulator is a Python-based application that simulates the operation of an elevator system. It is designed to help developers understand and test the logic behind elevator control algorithms.

Key features include:
- Simulation of multiple elevators
- Handling of concurrent requests
- Customizable number of floors and elevators
- Logging and debugging capabilities
- Extensible design for adding new features

## Setup

To set up the Elevator Simulator, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/elevator-simulator.git
    cd elevator-simulator
    ```

2. **Install Poetry** (if you haven't already):
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3. **Install dependencies**:
    ```sh
    poetry install
    ```

4. **Activate the virtual environment**:
    ```sh
    poetry shell
    ```

Now you are ready to use the Elevator Simulator.

## Usage


### Run a simulation

Use `elevator-sim`:

```
> elevator-sim --help
usage: elevator simulation [-h] [-f NUM_FLOORS] [-e NUM_ELEVATORS] [-c ELEVATOR_CAPACITIES] [-a {fcfs,scan,rl}] [-s SEED]

options:
  -h, --help            show this help message and exit
  -f NUM_FLOORS, --num_floors NUM_FLOORS
  -e NUM_ELEVATORS, --num_elevators NUM_ELEVATORS
  -c ELEVATOR_CAPACITIES, --elevator_capacities ELEVATOR_CAPACITIES
  -a {fcfs,scan,rl}, --agent_type {fcfs,scan,rl}
  -s SEED, --seed SEED
  ```

# ## 🚀 Training an Elevator RL Agent (PPO) using MLflow

This project leverages PPO (Proximal Policy Optimization) along with MLflow for detailed experiment tracking.

```
train_rl_agent \
    --num_floors 10 \
    --num_elevators 3 \
    --spawn_prob 0.2 \
    --total_timesteps 100000 \
    --checkpoint_freq 20000 \
    --model_path models/ppo_agent_full.pth
```

Track the experiment in MLflow UI:

```
poetry run mlflow ui
```