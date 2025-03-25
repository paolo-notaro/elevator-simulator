"""gae_utils.py: Generalized Advantage Estimation utilities."""

import torch


def compute_gae(
    rewards: list[torch.Tensor],
    values: list[torch.Tensor],
    next_value: torch.Tensor,
    dones: list[bool],
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
    advantages = torch.zeros(T, device=values[0].device)
    last_adv = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_adv = delta + gamma * lam * next_non_terminal * last_adv
    returns = advantages + values
    return advantages, returns
