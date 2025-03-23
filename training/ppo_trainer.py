import torch
from torch import optim
from torch.distributions import Categorical
import numpy as np
from tqdm import trange
import mlflow

from agents.rl import RLElevatorAgent
from environments.elevator import ElevatorAction
from environments.elevator_environment import ElevatorEnvironment
from environments.workload_scenario import RandomPassengerWorkloadScenario


class PPOTrainer:
    """Proximal Policy Optimization (PPO) trainer for the elevator environment."""

    def __init__(
        self,
        num_floors=10,
        num_elevators=3,
        embedding_dim=16,
        hidden_dim=128,
        lr=3e-4,
        clip_eps=0.2,
        gamma=0.99,
    ):
        """Initialize the PPO trainer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.num_actions = len(ElevatorAction)
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.clip_eps = clip_eps

        # Env
        self.env = ElevatorEnvironment(
            num_floors=num_floors,
            num_elevators=num_elevators,
            max_length=1000,
            workload_scenario=RandomPassengerWorkloadScenario(
                num_floors,
                spawn_prob=0.2,
                start_floor_probs=[1 / num_floors] * num_floors,
                end_floor_probs=[1 / num_floors] * num_floors,
            ),
            delay=0,  # no delay for training
        )

        # Agent
        self.agent = RLElevatorAgent(
            num_floors=num_floors,
            num_elevators=num_elevators,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        self.model = self.agent.model.to(self.device)
        self.embedding = self.agent.action_embedding.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def preprocess_observation(self, elevator_idx, obs):
        """Preprocess the observation for the given elevator."""
        pos = obs["elevators"]["current_floor"][elevator_idx] / self.num_floors
        load = obs["elevators"]["current_load"][elevator_idx] / 10
        up = obs["requests_up"].astype(float)
        down = obs["requests_down"].astype(float)
        return torch.tensor(np.concatenate([[pos, load], up, down]), dtype=torch.float32)

    def train_step(self):
        """Train the agent for a single episode."""
        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            actions = []
            log_probs = []
            values = []
            rewards = []

            prev_actions = []

            for idx in range(self.num_elevators):
                obs_tensor = self.preprocess_observation(idx, obs).to(self.device)
                obs_tensor = obs_tensor.unsqueeze(0)

                if prev_actions:
                    prev_tensor = torch.tensor(prev_actions, dtype=torch.long, device=self.device)
                    a_embed = self.embedding(prev_tensor)
                else:
                    a_embed = torch.zeros(self.embedding_dim, device=self.device)
                a_embed = a_embed.unsqueeze(0)

                action_logits, critic_value = self.model(obs_tensor, a_embed)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                actions.append(action.item())
                prev_actions.append(action.item())

                log_probs.append(log_prob)
                values.append(critic_value)
                rewards.append(0)  # filled later after step

            env_actions = [ElevatorAction(a) for a in actions]
            next_obs, reward, done, _, _ = self.env.step(env_actions)

            for r in rewards:
                r += reward / self.num_elevators  # shared reward

            # Compute advantage
            returns = sum(rewards)  # no GAE yet
            returns = torch.tensor(
                [returns] * self.num_elevators, dtype=torch.float32, device=self.device
            )

            values = torch.stack(values).squeeze()
            log_probs = torch.stack(log_probs)

            # Forward again for new log_probs
            new_log_probs = []
            prev_actions = []
            for idx in range(self.num_elevators):
                obs_tensor = self.preprocess_observation(idx, obs).to(self.device)
                if prev_actions:
                    prev_tensor = torch.tensor(prev_actions, dtype=torch.long, device=self.device)
                    a_embed = self.embedding(prev_tensor)
                else:
                    a_embed = torch.zeros(self.embedding_dim, device=self.device)

                logits, _ = self.model(obs_tensor, a_embed)
                dist = Categorical(logits=logits)
                new_log_probs.append(dist.log_prob(torch.tensor(actions[idx], device=self.device)))
                prev_actions.append(actions[idx])

            new_log_probs = torch.stack(new_log_probs)
            ratio = torch.exp(new_log_probs - log_probs)

            # PPO loss
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            policy_loss = -torch.min(ratio * returns, clipped * returns).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            obs = next_obs
            step += 1
            total_reward += reward

        return total_reward

    def train(self, episodes: int = 100):
        """Train the agent for a given number of episodes."""
        progress = trange(episodes, desc="Training", leave=True)
        for ep in progress:
            reward = self.train_step()
            progress.set_description(f"Episode {ep} | Reward: {reward:.2f}")
            mlflow.log_metric("episode_reward", reward, step=ep)
