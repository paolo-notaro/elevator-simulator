"""utils.py: Utility functions for elevator agents."""

import numpy as np


def unpack_flat_observation(obs, num_floors, num_elevators):
    obs = np.asarray(obs)

    floors = obs[:num_elevators]
    loads = obs[num_elevators : 2 * num_elevators]
    requests_up = obs[2 * num_elevators : 2 * num_elevators + num_floors]
    requests_down = obs[2 * num_elevators + num_floors :]

    return {
        "elevators": {
            "current_floor": (floors * num_floors).astype(int),
            "current_load": (loads * 10).astype(int),
        },
        "requests_up": requests_up.astype(bool),
        "requests_down": requests_down.astype(bool),
    }
