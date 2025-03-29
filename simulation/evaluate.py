"""evaluate.py: Evaluate the agents over multiple runs."""

from simulation.run import run_simulation, VALID_AGENTS, agent_type
from argparse import ArgumentParser, Namespace
import time

from statistics import stdev, mean
from tqdm import tqdm


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser("elevator evaluation")
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
    parser.add_argument("--model-path", type=str, default="models/ppo_agent_full.pth")
    parser.add_argument("-n", "--num_runs", type=int, default=100)
    args = parser.parse_args()

    args.agent_type = VALID_AGENTS[args.agent_type]

    return args


def evaluate(args: Namespace) -> None:
    """Evaluate the agent in the elevator environment."""

    rewards = []
    times = []

    args.delay = 0
    args.disable_prints = True
    avg_wait_time = 0
    avg_travel_time = 0
    avg_requests = 0
    for i in (progress_bar := tqdm(range(args.num_runs))):
        args.seed = i
        t_start = time.time()
        reward, total_requests, total_wait_time, total_travel_time = run_simulation(args)
        times.append(time.time() - t_start)
        rewards.append(reward)
        progress_bar.set_description(f"reward={reward:8.2f}")
        wait_time_i = total_wait_time / (total_requests + 1e-8)
        travel_time_i = total_travel_time / (total_requests + 1e-8)

        avg_wait_time += wait_time_i
        avg_travel_time += travel_time_i
        avg_requests += total_requests

    avg_wait_time /= args.num_runs
    avg_travel_time /= args.num_runs
    avg_requests /= args.num_runs

    print(
        f"\nReward min/avg/max (stdev): "
        f"{min(rewards):8.2f}/{mean(rewards):8.2f}/{max(rewards):8.2f} ({stdev(rewards):.2f})"
    )
    print(
        f"Average request wait time: {avg_wait_time:.2f},"
        f"average travel time: {avg_travel_time:.2f},"
        f"average requests completed: {avg_requests:.2f}"
    )
    print(f"(Total/average) simulation time: {sum(times):.2f}/{mean(times):.2f} seconds")


def main():
    """Main function."""
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
