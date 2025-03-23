"""train_rl.py: Train an RL agent using PPO and MLflow."""

import argparse
import os
import mlflow

from training.ppo_trainer import PPOTrainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("Train Elevator PPO Agent (custom implementation)")
    parser.add_argument(
        "--num_floors", type=int, default=10, help="Number of floors in the environment"
    )
    parser.add_argument("--num_elevators", type=int, default=1, help="Number of elevators")
    parser.add_argument(
        "--embedding_dim", type=int, default=16, help="Size of the action embedding"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden layer size of the model"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="Clipping epsilon for PPO loss")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ppo_agent_full.pth",
        help="Where to save model",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Train an RL agent using PPO and log results with MLflow."""
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    mlflow.set_experiment("PPO_Elevator_Agent")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "num_floors": args.num_floors,
                "num_elevators": args.num_elevators,
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "lr": args.lr,
                "episodes": args.episodes,
                "clip_eps": args.clip_eps,
                "gamma": args.gamma,
                "model_path": args.model_path,
                "lam": args.lam,
            }
        )

        # Create and initialize trainer and agent
        trainer = PPOTrainer(
            num_floors=args.num_floors,
            num_elevators=args.num_elevators,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            clip_eps=args.clip_eps,
            gamma=args.gamma,
            lam=args.lam,
        )

        # Train agent
        trainer.train(args.episodes)

        # Save model
        trainer.agent.save(f"{args.model_path}")
        mlflow.log_artifact(args.model_path)
        print(f"Model saved to {args.model_path}")


def main():
    """Main entry point function to train the RL agent."""
    arguments = parse_args()
    train(arguments)


if __name__ == "__main__":
    main()
