"""fcfs.py: First-Come, First-Served Elevator Agent."""

import numpy as np
from agents.base import BaseAgent
from environments.elevator import ElevatorAction
from environments.elevator_environment import ElevatorEnvironmentObservation
from agents.utils import unpack_flat_observation


class FCFSAgent(BaseAgent):
    """First-Come, First-Served Elevator Agent."""

    def __init__(self, num_floors: int, num_elevators: int):
        super().__init__(num_elevators=num_elevators, num_floors=num_floors)
        assert num_elevators == 1, "FCFS works clearly only with 1 elevator."

        # memory of request orders
        self.request_queue = []
        self.pending_request = None
        self.last_observation = None

    def _update_request_order(self, observation: ElevatorEnvironmentObservation):
        """Update the request order based on the current observation."""

        # Get the requests from the observation
        requests_up = observation["requests_up"]
        requests_down = observation["requests_down"]

        # Compare with last observation to extract new requests
        new_requests = []

        # XOR to get the new requests
        new_up = requests_up ^ self.last_observation["requests_up"]
        new_down = requests_down ^ self.last_observation["requests_down"]

        if np.any(new_up):
            new_requests.extend([int(x) for x in np.where(new_up)[0]])
        if np.any(new_down):
            new_requests.extend([int(x) for x in np.where(new_down)[0]])

        self.request_queue.extend(new_requests)

    def act(self, observation: ElevatorEnvironmentObservation) -> list[ElevatorAction]:
        """Act based on the observation."""
        if isinstance(observation, np.ndarray):  # SB3-style flat observation
            observation = unpack_flat_observation(
                observation, num_floors=self.num_floors, num_elevators=self.num_elevators
            )

        # Compare current observation with request order to update it
        if self.last_observation:
            self._update_request_order(observation)
        self.last_observation = observation

        if self.pending_request:
            # If there is a pending request, go to the floor
            next_floor = self.pending_request
        elif self.request_queue:
            # Get the next request and remove it from the list
            self.pending_request = self.request_queue.pop(0)
            next_floor = self.pending_request
        else:
            # No requests, go IDLE
            return [ElevatorAction.IDLE]

        # if floor is above, go UP, else go DOWN
        current_floor = observation["elevators"]["current_floor"][0]
        if next_floor > current_floor:
            return [ElevatorAction.UP]
        if next_floor < current_floor:
            return [ElevatorAction.DOWN]
        self.pending_request = None
        return [ElevatorAction.IDLE]
