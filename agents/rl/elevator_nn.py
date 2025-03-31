"""elevator_nn.py: Neural network for the elevator environment."""

import torch
import torch.nn as nn


class ElevatorActorCriticNetwork(nn.Module):
    """
    Actor-critic network for the elevator environment.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        use_batch_norm: bool = False,
        num_actions: int = 4,  # ElevatorAction.UP, DOWN, IDLE, STOP
    ):
        super().__init__()

        if use_batch_norm:
            layers = [
                nn.LayerNorm(input_dim + embedding_dim),
                nn.Linear(input_dim + embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ]
        else:
            layers = [nn.Linear(input_dim + embedding_dim, hidden_dim)]
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_prob))

        self.shared_net = nn.Sequential(*layers)

        self.actor_head = nn.Linear(hidden_dim, num_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, action_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.
        Args:
            obs: Observation tensor. Must be of size (batch_size, obs_dim).
            action_embed: Action embedding tensor. Must be of size (batch_size, embedding_dim).
        Returns:
            Tuple of action logits and critic value.

            action_logits: Tensor of size (batch_size, num_actions).
            critic_value: Tensor of size (batch_size, 1).
        """
        x = torch.cat([obs, action_embed], dim=-1)
        x = self.shared_net(x)
        return self.actor_head(x), self.critic_head(x)


class ElevatorActorCriticNetworkFloorWise(nn.Module):
    """
    Actor-critic network with per-floor shared MLP and aggregation.
    Suitable for variable number of floors.
    """

    def __init__(
        self,
        per_floor_dim: int,  # e.g. [internal_req, up_req, down_req, rel_floor_pos, is_current]
        scalar_dim: int,  # elevator-level features like load, time, etc.
        embedding_dim: int,
        floor_hidden_dim: int = 32,
        global_hidden_dim: int = 128,
        num_layers: int = 2,
        num_actions: int = 4,
        use_dropout: bool = False,
        dropout_prob: float = 0.3,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Shared MLP applied to each floor
        floor_layers = [
            nn.Linear(per_floor_dim, floor_hidden_dim),
            nn.ReLU(),
        ]
        if use_dropout:
            floor_layers.append(nn.Dropout(dropout_prob))
        self.floor_net = nn.Sequential(*floor_layers)

        # Final network combining aggregated floor info + elevator scalar + action embedding
        global_input_dim = floor_hidden_dim + scalar_dim + embedding_dim
        layers = []

        if use_batch_norm:
            layers.append(nn.LayerNorm(global_input_dim))
        layers.append(nn.Linear(global_input_dim, global_hidden_dim))
        if use_batch_norm:
            layers.append(nn.LayerNorm(global_hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(global_hidden_dim, global_hidden_dim))
            if use_batch_norm:
                layers.append(nn.LayerNorm(global_hidden_dim))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_prob))

        self.shared_net = nn.Sequential(*layers)

        self.actor_head = nn.Linear(global_hidden_dim, num_actions)
        self.critic_head = nn.Linear(global_hidden_dim, 1)

    def forward(
        self,
        per_floor_obs: torch.Tensor,  # shape: [batch_size, num_floors, per_floor_dim]
        scalar_obs: torch.Tensor,  # shape: [batch_size, scalar_dim]
        action_embed: torch.Tensor,  # shape: [batch_size, embedding_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_floors, _ = per_floor_obs.shape

        # Apply shared MLP to each floor
        floor_repr = self.floor_net(per_floor_obs.view(-1, per_floor_obs.shape[-1]))
        floor_repr = floor_repr.view(batch_size, num_floors, -1)

        # Aggregate floor representations (mean over floors)
        aggregated = floor_repr.mean(dim=1)

        # Combine with scalar features + action embedding
        combined = torch.cat([aggregated, scalar_obs, action_embed], dim=-1)

        # Shared core network
        x = self.shared_net(combined)

        return self.actor_head(x), self.critic_head(x)
