"""rl_agent.py: Reinforcement learning agent for the elevator environment."""

import torch
import numpy as np

from environments.elevator import ElevatorAction

from agents.base import BaseAgent
from agents.rl.action_embedding import ActionEmbedding
from agents.rl.elevator_nn import ElevatorActorCriticNetwork


class RLElevatorAgent(BaseAgent):
    """Reinforcement learning agent for the elevator environment (evaluation only)."""

    def __init__(
        self,
        num_floors: int,
        num_elevators: int,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_dropout: bool = True,
        dropout_prob: float = 0.3,
        use_batch_norm: bool = True,
        device: str = "cpu",
    ):
        super().__init__(num_floors=num_floors, num_elevators=num_elevators)

        self.num_actions = len(ElevatorAction)
        self.embedding_dim = embedding_dim
        self.device = device

        # Observation size = position (1) + load (1) + requests_up + requests_down
        self.obs_dim = num_floors * 2 + 2

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

    def _prepare_obs(self, elevator_idx, observation):
        elevator_obs = observation["elevators"]
        position = elevator_obs["current_floor"][elevator_idx] / self.num_floors
        load = elevator_obs["current_load"][elevator_idx] / 10

        requests_up = observation["requests_up"].astype(float)
        requests_down = observation["requests_down"].astype(float)

        obs_vec = np.concatenate([[position, load], requests_up, requests_down], axis=0)
        return torch.tensor(obs_vec, dtype=torch.float32).to(self.device)

    def act(self, observation) -> list[ElevatorAction]:
        """Returns a list of ElevatorAction (one for each elevator)."""

        actions = []
        prev_actions = []

        for elevator_idx in range(self.num_elevators):
            obs = self._prepare_obs(elevator_idx, observation)

            if prev_actions:
                prev_action_indices = torch.tensor(prev_actions, dtype=torch.long)
                action_embed = self.action_embedding(prev_action_indices).to(self.device)
            else:
                action_embed = torch.zeros(self.embedding_dim).to(self.device)

            with torch.no_grad():
                action_logits, _ = self.model(obs, action_embed)  # don't use critic value
                action_idx = torch.argmax(action_logits).cpu().item()

            actions.append(ElevatorAction(action_idx))
            prev_actions.append(action_idx)

        return actions

    def load(self, path: str):
        """Loads only the actor + shared network weights from PPO-trained model."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.action_embedding.load_state_dict(state["embedding"])
        self.action_embedding.eval()
        self.model.eval()

    def save(self, path: str):
        """Saves only the actor-relevant part of the model."""
        state = {"model": self.model.state_dict(), "embedding": self.action_embedding.state_dict()}
        torch.save(state, path)
