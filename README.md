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
usage: elevator simulation [-h] [-f NUM_FLOORS] [-e NUM_ELEVATORS] [-c ELEVATOR_CAPACITIES] [-a {fcfs,scan,rl}] [-s SEED] [-d DELAY] [--model-path MODEL_PATH]

options:
  -h, --help            show this help message and exit
  -f NUM_FLOORS, --num_floors NUM_FLOORS
  -e NUM_ELEVATORS, --num_elevators NUM_ELEVATORS
  -c ELEVATOR_CAPACITIES, --elevator_capacities ELEVATOR_CAPACITIES
  -a {fcfs,scan,rl}, --agent_type {fcfs,scan,rl}
  -s SEED, --seed SEED
  -d DELAY, --delay DELAY
  --model-path MODEL_PATH
  ```

# 🚀 Training an Elevator RL Agent (PPO) using MLflow

This project leverages PPO (Proximal Policy Optimization) along with MLflow for detailed experiment tracking.

```
train_rl_agent \
    --num_floors 10 \
    --num_elevators 3 \
    --embedding_dim 16 \
    --episodes 500 \
    --lr 3e-4 \
    --clip_eps 0.2 \
    --gamma 0.99 \
    --model_path models/ppo_agent_full.pth
```

Track the experiment in MLflow UI:

```
poetry run mlflow ui
```

## Evaluate an agent

```
python -m simulation.evaluate -n 100 -a rl
```

## Pre-train an agent with imitation learning

```
python -m training.imitation
```


# Results

episodes=1000, scenario= random workload scenario with default parameters

* FCFS: -1521.57 / -533.60 / 114.87 +- 263.92 / 0.19
* SCAN:  397.95 / 495.36 / 594.15 +- 39.50 / 0.18
* LOOK:  439.85 / 534.41 /  631.33 +- 37.01 / 0.18
* Best Imitation Learning model (teacher=LOOK, loss<0.2):  91.53/  528.36/  630.23 +- 41.63 / 0.77
* Best RL Model (after imitation teacher=LOOK, loss=0.2): 395.91/  503.18/  587.33 +- 40.07 / 0.65