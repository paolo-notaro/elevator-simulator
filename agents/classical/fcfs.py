"""fcfs.py: First-Come, First-Served Elevator Agent."""

from typing import Any

import numpy as np

from agents.base import BaseAgent

from environments.elevator import ElevatorAction
from environments.elevator_environment import ElevatorEnvironmentObservation


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
        """Track newly appearing floor requests using XOR trick."""
        new_up = observation["requests_up"] ^ self.last_observation["requests_up"]
        new_down = observation["requests_down"] ^ self.last_observation["requests_down"]

        new_requests = np.where(new_up | new_down)[0].tolist()
        self.request_queue.extend(new_requests)

    def act(
        self, observation: ElevatorEnvironmentObservation
    ) -> tuple[list[ElevatorAction], dict[str, Any]]:
        """Act based on the observation."""

        # Compare current observation with request order to update it
        if self.last_observation is not None:
            self._update_request_order(observation)
        self.last_observation = observation

        current_floor = observation["elevators"]["current_floor"][0]
        internal_requests = observation["elevators"]["internal_requests"][0]
        requests_up = observation["requests_up"]
        requests_down = observation["requests_down"]

        # Check if a stop is required
        if internal_requests[current_floor]:
            return [ElevatorAction.STOP], {}
        if requests_up[current_floor] or requests_down[current_floor]:
            return [ElevatorAction.STOP], {}

        if self.pending_request is not None:
            # If there is a pending request, go to the floor
            next_floor = self.pending_request
        elif self.request_queue:
            # Get the next request and remove it from the list
            self.pending_request = self.request_queue.pop(0)
            next_floor = self.pending_request
        else:
            # No external request in queue → check internal requests
            internal_targets = np.where(internal_requests)[0].tolist()
            if internal_targets:
                self.pending_request = internal_targets[0]  # serve first internal request
                next_floor = self.pending_request
            else:
                return [ElevatorAction.IDLE], {}

        # if floor is above, go UP, else go DOWN
        if next_floor > current_floor:
            return [ElevatorAction.UP], {}
        if next_floor < current_floor:
            return [ElevatorAction.DOWN], {}
        self.pending_request = None
        return [ElevatorAction.IDLE], {}
