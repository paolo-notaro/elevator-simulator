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
        "--entropy_coef", type=float, default=0.01, help="Coefficient for entropy loss"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/{run_name}_full.pth",
        help="Where to save model checkpoints",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Checkpoint model every n episodes, if model improves",
    )
    parser.add_argument(
        "--checkpoint-all",
        action="store_true",
        help="Checkpoint even if performance does not improve",
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
                "entropy_coef": args.entropy_coef,
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
            entropy_coef=args.entropy_coef,
        )

        # Train agent
        trainer.train(
            args.episodes,
            checkpoint_every=args.checkpoint_every,
            checkpoint_all=args.checkpoint_all,
        )

        # Save model
        final_model_path = args.model_path.format(run_name=mlflow.active_run().info.run_name)
        trainer.agent.save(final_model_path)
        mlflow.log_artifact(final_model_path)
        print(f"Final model saved to {final_model_path}.")


def main():
    """Main entry point function to train the RL agent."""
    arguments = parse_args()
    train(arguments)


if __name__ == "__main__":
    main()
