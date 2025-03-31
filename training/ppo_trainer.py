"""ppo_trainer.py: Proximal Policy Optimization trainer for the elevator environment."""

import torch
from torch import optim
from torch.distributions import Categorical
import numpy as np
from tqdm import trange
import mlflow

from environments.elevator_environment import ElevatorEnvironment
from environments.elevator import ElevatorAction
from agents.rl.rl_agent import RLElevatorAgent
from training.gae_utils import compute_gae


class PPOTrainer:
    """Proximal Policy Optimization trainer for the elevator environment."""

    def __init__(
        self,
        load_model_path: str = None,
        num_floors: int = 10,
        num_elevators: int = 3,
        max_episode_length: int = 1000,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        elevator_capacities: list[int] | int = 10,
        num_layers: int = 3,
        use_dropout: bool = False,
        dropout_prob: float = 0.3,
        use_batch_norm: bool = True,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        clip_value_loss_eps: float = 0.2,
    ):
        """Initialize the PPO trainer."""
        self.load_model_path = load_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.num_actions = len(ElevatorAction)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.clip_value_loss_eps = clip_value_loss_eps
        self.episode_count = 0

        self.env = ElevatorEnvironment(
            num_floors=num_floors,
            num_elevators=num_elevators,
            max_length=max_episode_length,
            workload_scenario=None,
            elevator_capacities=elevator_capacities,
            seed=42,
            delay=0,  # no delay for training
        )

        # Agent
        self.agent = RLElevatorAgent(
            num_floors=num_floors,
            num_elevators=num_elevators,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            elevator_capacities=elevator_capacities,
            dropout_prob=dropout_prob,
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
            device=self.device,
        )
        if load_model_path is not None:
            self.agent.load(load_model_path)

        # Model, optimizer, and action embedding
        self.optimizer = optim.Adam(
            list(self.agent.model.parameters()) + list(self.agent.action_embedding.parameters()),
            lr=lr,
        )

        # running reward average
        self.smoothed_reward = None  # Exponential average
        self.smoothing_alpha = 0.1  # Decay factor (tune if needed)

    def preprocess_observation(self, elevator_idx, obs):
        """Preprocess the observation for the given elevator."""
        pos = obs["elevators"]["current_floor"][elevator_idx] / self.num_floors
        load = obs["elevators"]["current_load"][elevator_idx] / 10
        up = obs["requests_up"].astype(float)
        down = obs["requests_down"].astype(float)
        return torch.tensor(np.concatenate([[pos, load], up, down]), dtype=torch.float32)

    def train_step(self, step_count: int):
        """Train the agent for a single episode."""
        obs, _ = self.env.reset()
        done = False
        total_reward = 0.0
        dones = []
        # Buffer per-elevator
        trajectories = [
            {"obs": [], "a_embed": [], "actions": [], "log_probs": [], "rewards": [], "values": []}
            for _ in range(self.num_elevators)
        ]

        warmup_steps = 100  # Warmup steps for critic head
        # Disable critic head training during warmup
        if step_count < warmup_steps:
            for param in self.agent.model.critic_head.parameters():
                param.requires_grad = False
                self.entropy_coef = 0.01
        else:
            for param in self.agent.model.critic_head.parameters():
                param.requires_grad = True
                self.entropy_coef = 0.1

        while not done:
            actions = []
            prev_actions = []

            for idx in range(self.num_elevators):
                obs_tensor = self.agent.prepare_observation(idx, obs).to(self.device).unsqueeze(0)

                if prev_actions:
                    prev_tensor = torch.tensor(prev_actions, dtype=torch.long, device=self.device)
                    action_embed = self.agent.action_embedding(prev_tensor).unsqueeze(0)
                else:
                    action_embed = torch.zeros((1, self.embedding_dim), device=self.device)

                logits, value = self.agent.model(obs_tensor, action_embed)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                actions.append(action.item())
                prev_actions.append(action.item())

                # Save trajectory step
                trajectory = trajectories[idx]
                trajectory["obs"].append(obs_tensor)
                trajectory["a_embed"].append(action_embed)
                trajectory["actions"].append(action)
                trajectory["log_probs"].append(log_prob)
                trajectory["values"].append(value.view(-1))  # scalar

            env_actions = [ElevatorAction(a) for a in actions]
            next_obs, reward, done, _, _ = self.env.step(env_actions)
            total_reward += reward
            dones.append(done)

            # Shared reward — add same reward to each elevator
            for trajectory in trajectories:
                trajectory["rewards"].append(
                    torch.tensor(
                        reward / self.num_elevators, dtype=torch.float32, device=self.device
                    )
                )

            obs = next_obs

        # Post-episode update
        policy_loss = 0.0
        value_loss = 0.0
        elevator_entropies = []
        elevator_advantages = []
        prev_actions = []
        for idx in range(self.num_elevators):
            trajectory = trajectories[idx]

            # Elevator forward pass
            with torch.no_grad():
                obs_tensor = self.agent.prepare_observation(idx, obs).to(self.device).unsqueeze(0)

                # compose previous action (other elevators, same timestep) embedding
                if prev_actions:
                    prev_tensor = torch.tensor(prev_actions, dtype=torch.long, device=self.device)
                    action_embed = self.agent.action_embedding(prev_tensor).unsqueeze(0)
                else:
                    action_embed = torch.zeros((1, self.embedding_dim), device=self.device)

                # compute elevator action and critic value
                action_logits, next_value = self.agent.model(obs_tensor, action_embed)
                action_idx = torch.argmax(action_logits).cpu().item()
                prev_actions.append(action_idx)
                next_value = next_value.squeeze(0)

            # Estimate advantages and returns using GAE
            values_tensor = torch.stack(trajectory["values"]).view(-1).to(self.device)
            advantages, returns = compute_gae(
                rewards=trajectory["rewards"],
                values=values_tensor,
                next_value=next_value,
                dones=dones,
                gamma=self.gamma,
                lam=self.lam,
            )
            assert (
                returns.shape == values_tensor.shape
            ), f"Mismatch: {returns.shape} vs {values_tensor.shape}"

            # Critic loss (MSE between predicted values and returns)
            # value_loss += torch.nn.functional.mse_loss(values_tensor, returns.detach())
            value_pred_clipped = values_tensor + (returns - values_tensor).clamp(
                -self.clip_value_loss_eps, self.clip_value_loss_eps
            )
            value_loss_clipped = (value_pred_clipped - returns).pow(2)
            value_loss_unclipped = (values_tensor - returns).pow(2)
            value_loss += 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            # Policy loss (PPO clipped surrogate objective)
            log_probs_old = torch.stack(trajectory["log_probs"])
            actions = torch.stack(trajectory["actions"])
            obs_batch = torch.cat(trajectory["obs"])
            embed_batch = torch.cat(trajectory["a_embed"])

            logits, _ = self.agent.model(obs_batch, embed_batch)
            dist = Categorical(logits=logits)
            log_probs_new = dist.log_prob(actions)
            ratio = torch.exp(log_probs_new - log_probs_old)
            entropy = dist.entropy().mean()

            adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # adv = torch.clamp(adv, -5.0, 5.0)

            clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv

            policy_loss += -torch.min(ratio * adv, clip_adv).mean() - self.entropy_coef * entropy

            elevator_entropies.append(entropy.item())
            elevator_advantages.append(adv.mean().item())

        # compute total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss
        total_loss /= self.num_elevators

        # back-propagation
        self.optimizer.zero_grad()
        total_loss.backward()

        # Clip gradient norms to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_norm=1.0)

        # Update step
        self.optimizer.step()

        # Report values on MLFlow
        mlflow.log_metric("policy_loss", policy_loss.item(), step=self.episode_count)
        mlflow.log_metric("value_loss", value_loss.item(), step=self.episode_count)
        mlflow.log_metric("total_loss", total_loss.item(), step=self.episode_count)
        mlflow.log_metric(
            "mean_elevator_entropy", np.mean(elevator_entropies), step=self.episode_count
        )
        mlflow.log_metric("mean_advantage", np.mean(elevator_advantages), step=self.episode_count)
        mlflow.log_metric(
            "value_vs_return_diff",
            (values_tensor - returns).abs().mean().item(),
            step=self.episode_count,
        )

        total_grad_norm = sum(
            p.grad.norm().item() for p in self.agent.model.parameters() if p.grad is not None
        )
        mlflow.log_metric("grad_norm", total_grad_norm, step=self.episode_count)

        return total_reward

    def train(
        self, episodes: int = 1000, checkpoint_every: int = 100, checkpoint_all: bool = False
    ):
        """Train the agent for a given number of episodes."""
        self.agent.model.train()
        self.agent.action_embedding.train()

        progress = trange(episodes, desc="Training", leave=True)
        best_smoothed_reward = -np.inf
        for self.episode_count in progress:
            reward = self.train_step(step_count=self.episode_count)

            if self.smoothed_reward is None:
                self.smoothed_reward = reward
            self.smoothed_reward = (
                self.smoothing_alpha * reward + (1 - self.smoothing_alpha) * self.smoothed_reward
            )

            progress.set_description(
                f"Episode {self.episode_count + 1:4d} | R: {reward:7.2f} | "
                f"Smoothed R: {self.smoothed_reward:7.2f}"
            )
            mlflow.log_metric("episode_reward", reward, step=self.episode_count)
            mlflow.log_metric("smoothed_reward", self.smoothed_reward, step=self.episode_count)

            if (self.episode_count + 1) % checkpoint_every == 0:
                if self.smoothed_reward > best_smoothed_reward or checkpoint_all:
                    print("Saving best model...")
                    self.agent.save(
                        f"models/{mlflow.active_run().info.run_name}_{self.episode_count + 1}.pth"
                    )
                    best_smoothed_reward = self.smoothed_reward

        self.agent.model.eval()
        self.agent.action_embedding.eval()
