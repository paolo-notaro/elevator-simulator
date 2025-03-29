"""rl_agent.py: Reinforcement learning agent for the elevator environment."""

from typing import Any

import numpy as np
import torch.nn.functional as F
import torch

from agents.base import BaseAgent
from agents.rl.action_embedding import ActionEmbedding
from agents.rl.elevator_nn import ElevatorActorCriticNetwork

from environments.elevator import ElevatorAction


class RLElevatorAgent(BaseAgent):
    """Reinforcement learning agent for the elevator environment (evaluation only)."""

    def __init__(
        self,
        num_floors: int,
        num_elevators: int,
        elevator_capacities: list[int] | int = 10,  # default elevator capacity
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_dropout: bool = False,
        dropout_prob: float = 0.3,
        use_batch_norm: bool = False,
        device: str = "cpu",
    ):
        super().__init__(num_floors=num_floors, num_elevators=num_elevators)

        self.num_actions = len(ElevatorAction)
        if isinstance(elevator_capacities, int):
            elevator_capacities = [elevator_capacities] * num_elevators
        elif isinstance(elevator_capacities, list):
            assert (
                len(elevator_capacities) == num_elevators
            ), "Number of elevators and elevator capacities do not match"
        else:
            raise ValueError("Invalid elevator capacities")
        self.elevator_capacities = elevator_capacities
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.embedding_dim = embedding_dim

        self.device = device

        self.action_embedding = ActionEmbedding(self.num_actions, embedding_dim).to(device)
        self.model = ElevatorActorCriticNetwork(
            input_dim=self.obs_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_actions=self.num_actions,
            use_dropout=use_dropout,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm,
        ).to(device)
        self.model.eval()
        self.action_embedding.eval()

    @property
    def obs_dim(self) -> int:
        """Observation dimension for the elevator agent."""
        # [position, load] + position_one_hot + internal requests + requests up + requests down
        return 2 + 4 * self.num_floors

    def prepare_observation(self, elevator_idx, observation):
        elevator_obs = observation["elevators"]

        # One-hot encode position
        position = elevator_obs["current_floor"][elevator_idx]
        position_one_hot = np.zeros(self.num_floors, dtype=float)
        position_one_hot[position] = 1.0

        load = elevator_obs["current_load"][elevator_idx] / self.elevator_capacities[elevator_idx]
        internal_requests = elevator_obs["internal_requests"][elevator_idx]

        requests_up = observation["requests_up"].astype(float)
        requests_down = observation["requests_down"].astype(float)

        obs_vec = np.concatenate(
            [position_one_hot, [position, load], internal_requests, requests_up, requests_down],
            axis=0,
        )
        return torch.tensor(obs_vec, dtype=torch.float32).to(self.device)

    def act(
        self, observation, stochastic: bool = True
    ) -> tuple[list[ElevatorAction], dict[str, Any]]:
        """
        Selects actions for all elevators based on the observation.

        Args:
            observation: Observation from the environment.
            stochastic: Whether to sample actions stochastically.

        Returns:
            Tuple of actions and additional information.
        """

        actions = []
        prev_actions = []

        action_logits = []
        critic_values = []
        for elevator_idx in range(self.num_elevators):
            obs = self.prepare_observation(elevator_idx, observation)

            if prev_actions:
                prev_action_indices = torch.tensor(prev_actions, dtype=torch.long)
                action_embed = self.action_embedding(prev_action_indices).to(self.device)
            else:
                action_embed = torch.zeros(self.embedding_dim).to(self.device)

            with torch.no_grad():
                action_logits_i, critic_value = self.model(
                    obs, action_embed
                )  # don't use critic value

                # pick action
                if stochastic:
                    action_probs = F.softmax(action_logits_i, dim=-1)
                    action_distribution = torch.distributions.Categorical(action_probs)
                    action_idx = action_distribution.sample().item()
                else:
                    action_idx = torch.argmax(action_logits_i).cpu().item()

            actions.append(ElevatorAction(action_idx))
            action_logits.append(action_logits_i.cpu().numpy())
            prev_actions.append(action_idx)
            critic_values.append(critic_value.item())

        return actions, {
            "action_logits": action_logits,
            "critic_values": critic_values,
        }

    def load(self, path: str):
        """Loads only the actor + shared network weights from PPO-trained model."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model"])
        self.action_embedding.load_state_dict(state["embedding"])
        self.action_embedding.eval()
        self.model.eval()

    def save(self, path: str):
        """Saves only the actor-relevant part of the model."""
        state = {"model": self.model.state_dict(), "embedding": self.action_embedding.state_dict()}
        torch.save(state, path)
