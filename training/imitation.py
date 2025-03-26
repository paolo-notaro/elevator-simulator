"""imitation.py: Train the RL agent by imitation learning."""

import torch
import numpy as np
from agents.rl.rl_agent import RLElevatorAgent
from agents.base import BaseAgent
from agents.classical.look import LOOKAgent
from environments.elevator_environment import ElevatorEnvironment


def train_by_imitation(
    rl_agent: RLElevatorAgent,
    imitation_agent: BaseAgent,
    env: ElevatorEnvironment,
    num_episodes: int,
    batch_size: int = 32,
):
    """Train the RL agent by imitation learning.

    Args:
        rl_agent: The RL agent to train.
        imitation_agent: The agent to imitate.
        num_episodes: The number of episodes to train for.
    """
    rl_agent.action_embedding.train()
    rl_agent.model.train()

    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rl_agent.model.parameters(), lr=3e-4)
    loss_value = np.inf
    for episode in range(num_episodes):
        # Reset the environment
        observation, _ = env.reset(seed=episode)
        done = False

        obs_batch, targets_batch = [], []

        while not done:
            # Get the action from the imitation agent (ground truth)
            actions, _ = imitation_agent.act(observation)
            imitation_action = torch.tensor(actions[0].value, dtype=torch.long)

            # Get the action from the RL agent
            obs_tensor = rl_agent.prepare_observation(0, observation)

            # append to inputs and targets to batch
            obs_batch.append(obs_tensor.unsqueeze(0))
            targets_batch.append(imitation_action.unsqueeze(0))

            # Take the action in the environment
            observation, reward, done, _, __ = env.step(actions)

            # Update the weights
            if len(obs_batch) >= batch_size or done:

                action_embed = torch.zeros((len(obs_batch), rl_agent.embedding_dim)).to(
                    rl_agent.device
                )
                obs_batch = torch.cat(obs_batch, dim=0).to(rl_agent.device)
                targets_batch = torch.cat(targets_batch, dim=0).to(rl_agent.device)

                # Forward pass
                actions_batch, _ = rl_agent.model(obs_batch, action_embed)  # don't use critic value

                # Compute the loss
                loss = loss_criterion(actions_batch, targets_batch)
                optimizer.zero_grad()  # zero the gradients
                loss.backward()  # compute the gradients
                optimizer.step()  # update the weights

                loss_value = loss.item()
                print(f"Episode {episode + 1}, Step: {env.step_count}, Loss: {loss_value}")
                obs_batch, targets_batch = [], []

        if loss_value <= 0.1:
            break

    rl_agent.action_embedding.eval()
    rl_agent.model.eval()
    rl_agent.save("models/imitation_model.pth")


def main():
    """Train the RL agent by imitation learning."""
    # Create the environment
    env = ElevatorEnvironment(num_floors=10, num_elevators=1, max_length=96)

    # Create the RL agent
    rl_agent = RLElevatorAgent(
        num_floors=10,
        num_elevators=1,
        embedding_dim=16,
        hidden_dim=128,
        num_layers=3,
        use_dropout=False,
        dropout_prob=0.3,
        use_batch_norm=False,
        device=torch.device("cuda"),
    )

    # Create the imitation agent
    imitation_agent = LOOKAgent(num_floors=10, num_elevators=1)

    # Train the RL agent by imitation learning
    train_by_imitation(rl_agent, imitation_agent, env, num_episodes=100)


if __name__ == "__main__":
    main()
