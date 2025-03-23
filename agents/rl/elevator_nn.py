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
        num_actions: int = 3,  # ElevatorAction.UP, DOWN, IDLE
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim + embedding_dim)

        layers = [nn.Linear(input_dim + embedding_dim, hidden_dim)]
        if use_batch_norm:
            layers.append(nn.LayerNorm(hidden_dim))
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

    def forward(self, obs: torch.Tensor, action_embed: torch.Tensor):
        x = torch.cat([obs, action_embed], dim=-1)
        x = self.input_norm(x)
        x = self.shared_net(x)
        return self.actor_head(x), self.critic_head(x)
