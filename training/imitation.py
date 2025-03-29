"""imitation.py: Train the RL agent by imitation learning."""

import torch
import torch.nn.functional as F
import mlflow

from agents.rl.rl_agent import RLElevatorAgent
from agents.base import BaseAgent
from agents.classical.look import LOOKAgent
from environments.elevator_environment import ElevatorEnvironment, ElevatorAction


def evaluate_student(
    agent: RLElevatorAgent,
    env: ElevatorEnvironment,
    step: int,
    num_episodes: int = 10,
):
    """Evaluate the student agent in the environment."""
    agent.model.eval()
    agent.action_embedding.eval()

    all_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=None)
        done = False
        total_reward = 0.0

        while not done:
            prev_actions = []
            actions = []

            for idx in range(agent.num_elevators):
                obs_tensor = agent.prepare_observation(idx, obs).unsqueeze(0).to(agent.device)

                if prev_actions:
                    prev_tensor = torch.tensor(prev_actions, dtype=torch.long, device=agent.device)
                    action_embed = agent.action_embedding(prev_tensor).unsqueeze(0)
                else:
                    action_embed = torch.zeros((1, agent.embedding_dim), device=agent.device)

                logits, _ = agent.model(obs_tensor, action_embed)
                action = torch.argmax(logits, dim=-1).item()
                prev_actions.append(action)
                actions.append(ElevatorAction(action))

            obs, reward, done, _, _ = env.step(actions)
            total_reward += reward

        all_rewards.append(total_reward)
        print(f"Evaluation Episode {ep + 1}: reward = {total_reward:.2f}")

    mean_reward = sum(all_rewards) / num_episodes
    mlflow.log_metric("episode_reward.student", mean_reward, step=step)
    print(
        f"Student Evaluation over {num_episodes} episodes: "
        f"mean = {mean_reward:.2f}, min = {min(all_rewards):.2f}, max = {max(all_rewards):.2f}"
    )
    agent.model.train()
    agent.action_embedding.train()


def discounted_cumsum(rewards: list[float], gamma: float) -> torch.Tensor:
    """Compute the discounted cumulative sum of rewards.
    Args:
        rewards: List of rewards.
        gamma: Discount factor.
    Returns:
        Tensor of discounted cumulative sum of rewards.
    """
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32, requires_grad=False)


def episode_loop(
    rl_agent: RLElevatorAgent,
    imitation_agent: BaseAgent,
    env: ElevatorEnvironment,
    episode_count: int,
    batch_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Run a single episode loop for the RL agent and the imitation agent.
    Args:
        rl_agent: The RL agent to train.
        imitation_agent: The agent to imitate.
        env: The elevator environment.
        episode_count: The number of episodes to run.
        batch_size: The batch size for training.

    Returns:
        obs_batches: List of observation batches.
        imitation_action_batches: List of imitation action batches.
        action_embed_batches: List of action embedding batches.
        rewards: List of rewards.
    """
    # Reset the environment
    print(f"Starting episode {episode_count + 1}...")
    observation, _ = env.reset(seed=episode_count)
    done = False
    obs_batches, imitation_action_batches, action_embed_batches = [], [], []
    rewards = []

    obs_batch, imitation_action_batch = [], []
    while not done:
        # Get the action from the imitation agent (ground truth)
        actions, _ = imitation_agent.act(observation)
        imitation_action = actions[0].value  # single elevator

        # Prepare the input for the RL agent
        obs_tensor = rl_agent.prepare_observation(0, observation)

        # Append inputs and targets to batch
        obs_batch.append(obs_tensor.unsqueeze(0))
        imitation_action_batch.append(imitation_action)

        # Take the action in the environment
        observation, reward, done, _, __ = env.step(actions)

        # Append reward
        rewards.append(reward)

        # print(
        #    f"Episode {episode_count + 1}, step: {env.step_count},"
        #    f" imitation action: {imitation_action}, reward: {reward}"
        # )

        # Batch creation (after BATCH_SIZE steps or at the end of an episode)
        if len(obs_batch) >= batch_size or done:

            action_embed = torch.zeros(
                (len(obs_batch), rl_agent.embedding_dim), device=rl_agent.device
            )
            obs_batch = torch.cat(obs_batch, dim=0).to(rl_agent.device)
            imitation_action_batch = [
                torch.tensor(action, dtype=torch.long, requires_grad=False)
                for action in imitation_action_batch
            ]
            imitation_action_batch = torch.stack(
                imitation_action_batch,
                dim=0,
            ).to(rl_agent.device)

            # Append to batch lists
            obs_batches.append(obs_batch)
            imitation_action_batches.append(imitation_action_batch)
            action_embed_batches.append(action_embed)

            # print(f"Batch created with size {len(obs_batch)}")

            obs_batch, imitation_action_batch = [], []

    print(
        f"Finished episode {episode_count + 1}, total reward: {sum(rewards):.2f},"
        f" total steps: {env.step_count}"
    )

    return obs_batches, imitation_action_batches, action_embed_batches, rewards


def train_by_imitation(
    rl_agent: RLElevatorAgent,
    imitation_agent: BaseAgent,
    env: ElevatorEnvironment,
    num_episodes: int,
    batch_size: int = 32,
    value_loss_coef: float = 0.5,
    gamma: float = 0.99,
    learning_rate: float = 3e-4,
    early_stop_loss: float = 0.05,
    eval_every: int = 10,
    eval_episodes: int = 10,
    save_every: int = 100,
):
    """Train the RL agent by imitation learning.

    Args:
        rl_agent: The RL agent to train.
        imitation_agent: The agent to imitate.
        num_episodes: The number of episodes to train for.
        batch_size: The batch size for training.
        value_loss_coef: The coefficient for the value loss.
        gamma: The discount factor for the rewards.
        learning_rate: The learning rate for the optimizer.
        early_stop_loss: The early stopping loss threshold.
        eval_every: The number of episodes between evaluations.
        eval_episodes: The number of episodes to evaluate the agent.
        save_every: The number of episodes between model saves.
    """
    rl_agent.action_embedding.train()
    rl_agent.model.train()

    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rl_agent.model.parameters(), lr=learning_rate)
    for episode in range(num_episodes):

        # Episode loop
        obs_batches, imitation_action_batches, action_embed_batches, rewards = episode_loop(
            rl_agent, imitation_agent, env, episode, batch_size
        )

        # Training steps (after each episode)
        print(f"Starting training step of episode={episode + 1}...")

        # Compute the discounted cumulative sum of rewards
        returns = discounted_cumsum(rewards, gamma)
        return_start_index = 0

        avg_episode_loss = 0.0
        for b, (action_embed_batch, obs_batch, imitation_action_batch) in enumerate(
            zip(action_embed_batches, obs_batches, imitation_action_batches)
        ):

            # Forward pass
            actions_batch, critic_batch = rl_agent.model(obs_batch, action_embed_batch)

            # Compute the loss
            actor_loss = loss_criterion(actions_batch, imitation_action_batch)
            returns_batch = (
                returns[return_start_index : return_start_index + obs_batch.size(0)]
                .to(rl_agent.device)
                .detach()
            )
            value_loss = F.mse_loss(critic_batch.squeeze(-1), returns_batch)
            total_loss = actor_loss + value_loss_coef * value_loss

            # Gradient update step
            optimizer.zero_grad()  # zero the gradients
            total_loss.backward()  # compute the gradients
            optimizer.step()  # update the weights

            actor_loss = actor_loss.item()
            value_loss = value_loss.item()
            total_loss = total_loss.item()
            avg_episode_loss += total_loss
            print(
                f"Episode {episode + 1:4d}, Batch: {b + 1:3d}, Loss: {total_loss:7.4f}"
                f" (actor: {actor_loss:7.4f} + value: {value_loss:7.4f})"
            )
            mlflow.log_metric("actor_loss", actor_loss, step=episode * 100 + b)
            mlflow.log_metric("value_loss", value_loss, step=episode * 100 + b)
            mlflow.log_metric("total_loss", total_loss, step=episode * 100 + b)

            return_start_index += len(obs_batch)

        # Report the average loss for the episode
        avg_episode_loss /= len(obs_batches)
        print(
            f"Finished training for episode {episode + 1},"
            f" average episode loss: {avg_episode_loss:7.4f}"
        )
        mlflow.log_metric("avg_episode_loss", avg_episode_loss, step=episode + 1)
        mlflow.log_metric("episode_reward.teacher", sum(rewards), step=episode + 1)

        if (episode + 1) % eval_every == 0:
            # Evaluate the student agent
            print(f"Evaluating student agent at episode {episode + 1}...")
            evaluate_student(
                rl_agent,
                env,
                step=episode + 1,
                num_episodes=eval_episodes,
            )

        if (episode + 1) % save_every == 0:
            print(f"Saving model at episode {episode + 1}...")
            rl_agent.save(f"models/imitation_model_{episode + 1}.pth")

        # Early stopping condition
        if avg_episode_loss <= early_stop_loss:
            break

    rl_agent.save("models/imitation_model.pth")


def main():
    """Train the RL agent by imitation learning."""

    # Inputs and hyperparameters
    num_floors = 10
    num_elevators = 1
    max_episode_length = 1000
    embedding_dim = 16
    hidden_dim = 128
    num_layers = 3
    use_dropout = False
    dropout_prob = 0.3
    use_batch_norm = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 1000
    batch_size = 32
    value_loss_coef = 0.25
    gamma = 0.9
    learning_rate = 1e-4
    early_stop_loss = 0.05
    eval_every = 10
    eval_episodes = 10
    save_every = 100

    # Create the environment
    env = ElevatorEnvironment(
        num_floors=num_floors, num_elevators=num_elevators, max_length=max_episode_length, delay=0
    )

    # Create the RL agent
    rl_agent = RLElevatorAgent(
        num_floors=num_floors,
        num_elevators=num_elevators,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_dropout=use_dropout,
        dropout_prob=dropout_prob,
        use_batch_norm=use_batch_norm,
        device=device,
    )

    # Create the imitation agent
    imitation_agent = LOOKAgent(num_floors=num_floors, num_elevators=num_elevators)

    # Train the RL agent by imitation learning
    mlflow.set_experiment("ImitationLearning")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "num_floors": num_floors,
                "num_elevators": num_elevators,
                "max_episode_length": max_episode_length,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "use_dropout": use_dropout,
                "dropout_prob": dropout_prob,
                "use_batch_norm": use_batch_norm,
                "device": device.type,
                "num_episodes": num_episodes,
                "batch_size": batch_size,
                "value_loss_coef": value_loss_coef,
                "gamma": gamma,
                "learning_rate": learning_rate,
                "early_stop_loss": early_stop_loss,
                "eval_every": eval_every,
                "eval_episodes": eval_episodes,
                "save_every": save_every,
            }
        )
        train_by_imitation(
            rl_agent,
            imitation_agent,
            env,
            num_episodes=num_episodes,
            batch_size=batch_size,
            value_loss_coef=value_loss_coef,
            gamma=gamma,
            learning_rate=learning_rate,
            early_stop_loss=early_stop_loss,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            save_every=save_every,
        )


if __name__ == "__main__":
    main()
