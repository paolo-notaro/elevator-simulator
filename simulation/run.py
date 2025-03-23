from argparse import ArgumentParser, ArgumentTypeError, Namespace
import torch

from agents.base import BaseAgent
from agents.classical.fcfs import FCFSAgent
from agents.classical.scan import SCANAgent
from agents.rl import RLElevatorAgent
from environments.elevator_environment import ElevatorEnvironment

VALID_AGENTS = {"fcfs": FCFSAgent, "scan": SCANAgent, "rl": RLElevatorAgent}


def agent_type(name: str) -> type[BaseAgent]:
    """Return the agent class for the given name."""
    name_ = name.lower()
    if name_ not in VALID_AGENTS:
        raise ArgumentTypeError(f"Invalid agent type: {name}")
    return name_


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser("elevator simulation")
    parser.add_argument("-f", "--num_floors", type=int, default=10)
    parser.add_argument("-e", "--num_elevators", type=int, default=1)
    parser.add_argument("-c", "--elevator_capacities", type=int, default=8)
    parser.add_argument(
        "-a",
        "--agent_type",
        type=agent_type,
        choices=list(VALID_AGENTS.keys()),
        default="scan",
    )
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-d", "--delay", type=float, default=1.0)
    parser.add_argument("--model-path", type=str, default="models/ppo_agent_full.pth")
    parser.add_argument("--disable_prints", type=bool, default=False)
    args = parser.parse_args()

    args.agent_type = VALID_AGENTS[args.agent_type]

    return args


def run_simulation(args: Namespace):
    """Run a simulation of the elevator environment."""
    env = ElevatorEnvironment(
        num_elevators=args.num_elevators,
        num_floors=args.num_floors,
        workload_scenario=None,
        elevator_capacities=args.elevator_capacities,
        seed=args.seed,
        delay=args.delay,
    )
    observation = env.reset()

    if args.num_elevators > 1 and args.agent_type in (FCFSAgent, SCANAgent):
        raise ValueError("FCFS and SCAN agents only support single elevators")

    agent = args.agent_type(num_elevators=args.num_elevators, num_floors=args.num_floors)
    if args.agent_type == RLElevatorAgent:
        agent.load(args.model_path)

    observation, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        if not args.disable_prints:
            print(f"\n=======================\nStep {env.step_count}")
            print("Observation:", observation)
        actions = agent.act(observation)

        observation, reward, done, truncated, info = env.step(actions)

        if not args.disable_prints:
            print("Actions:", actions)
            print("Reward:", reward)
            print("Hidden state:", env.passenger_requests)
        total_reward += reward

    if not args.disable_prints:
        print(f"Simulation ended, total reward: {total_reward}")

    return total_reward


def main():
    """Run the simulation."""
    args = parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
