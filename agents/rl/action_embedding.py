"""action_embedding.py: Module for action embeddings."""

import torch
import torch.nn as nn


class ActionEmbedding(nn.Module):
    """
    Embedding model for actions.

    This model embeds a sequence of actions into a single vector, to be used as input to
      another neural network.
    """

    def __init__(self, num_actions: int, embedding_dim: int = 16):
        """
        Initialize the action embedding model.

        Args:
            num_actions: The number of different action types ("dictionary size").
            embedding_dim: The dimension of the embedding.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embedding_dim)

    def forward(self, action_indices: list[int]) -> torch.Tensor:
        """
        Forward pass of the action embedding model.
        """
        embedded = self.embedding(action_indices)
        aggregated = embedded.mean(dim=0)
        return aggregated
