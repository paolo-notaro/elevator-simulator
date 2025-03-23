"""gae_utils.py: Generalized Advantage Estimation utilities."""

import torch


def compute_gae(
    rewards: list[torch.tensor],
    values: list[torch.tensor],
    next_value: float,
    gamma: float,
    lam: float,
):
    """
    Compute GAE advantages and returns for a trajectory.
    Args:
        rewards: (T,) list of rewards
        values: (T,) list of state values
        next_value: scalar value of last state
        gamma: discount factor
        lam: GAE lambda
    Returns:
        advantages: (T,) tensor
        returns: (T,) tensor
    """
    T = len(rewards)
    advantages = torch.zeros(T).to(values.device)
    last_adv = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_adv = delta + gamma * lam * last_adv
        next_value = values[t]
    returns = advantages + values
    return advantages, returns
