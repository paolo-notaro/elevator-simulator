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
Reward min /mean / max (std) / wait_time / travel_time / requests_served / avg_run_duration

* FCFS:  -3260.21/ -204.63/  430.82 (905.86) / 16.33  / 10.82 / 149.76 / 0.17
* SCAN:  -2197.31/  402.73/  492.52 (108.74) / 11.79 /  7.45 / 195.67 / 0.15
* LOOK:  -1546.42/  413.51/  512.16 (121.87) / 9.89 /  6.47 / 196.40 / 0.18
* Best Imitation Learning model (teacher=LOOK, loss<0.2):  -2788.44/  350.07/  512.70 (354.19) / 10.14 / 6.74 / 191.27 / 3.00
* Best RL Model (after imitation teacher=LOOK, loss=0.2): -2169.88/  402.35/  500.48 (106.49) / 11.35 / 7.49 / 196.00 / 3.11