"""scan.py: SCAN elevator agent."""

from enum import Enum
from typing import Any

import numpy as np

from agents.base import BaseAgent
from environments.elevator import ElevatorAction
from environments.elevator_environment import ElevatorEnvironmentObservation


class SCANDirection(Enum):
    """Enumeration for the direction of the SCAN algorithm."""

    UP = 1
    DOWN = -1


class SCANAgent(BaseAgent):
    def __init__(self, num_floors: int, num_elevators: int):
        """Initialize the SCAN agent.

        Args:
            num_floors: The number of floors in the building.
            num_elevators: The number of elevators in the building
        """
        super().__init__(num_elevators=num_elevators, num_floors=num_floors)
        assert num_elevators == 1, "SCAN is designed for a single elevator."
        self.direction = SCANDirection.UP

    def act(
        self, observation: ElevatorEnvironmentObservation
    ) -> tuple[list[ElevatorAction], dict[str, Any]]:
        """Act based on the observation.

        Args:
            observation: The observation from the environment.

        Returns:
            A tuple containing the action(s) to take and additional information.
            A dictionary with additional information about the action taken.
        """

        elevator_position = observation["elevators"]["current_floor"][0]
        internal_requests = observation["elevators"]["internal_requests"][0]
        requests_up = observation["requests_up"]
        requests_down = observation["requests_down"]

        stop_needed = self._should_stop(
            pos=elevator_position,
            internal_requests=internal_requests,
            requests_up=requests_up,
            requests_down=requests_down,
        )

        if stop_needed:
            action = ElevatorAction.STOP
        else:
            action = self._move(elevator_position)

        return [action], {}

    def _should_stop(
        self,
        pos: int,
        internal_requests: np.ndarray,
        requests_up: np.ndarray,
        requests_down: np.ndarray,
    ) -> bool:
        """Determine if the elevator should STOP at the current floor."""

        if internal_requests[pos]:
            return True

        if self.direction == SCANDirection.UP:
            if requests_up[pos]:
                return True
            # Edge: If we're at the top floor and need to reverse, check for DOWN requests
            if pos == self.num_floors - 1 and requests_down[pos]:
                return True

        if self.direction == SCANDirection.DOWN:
            if requests_down[pos]:
                return True
            if pos == 0 and requests_up[pos]:
                return True

        return False

    def _move(self, pos: int) -> ElevatorAction:
        """Move in the current direction. Reverse only when stuck."""
        if self.direction == SCANDirection.UP:
            if pos < self.num_floors - 1:
                return ElevatorAction.UP
            else:
                self.direction = SCANDirection.DOWN
                return ElevatorAction.IDLE  # Let next act() call handle movement after reversal

        elif self.direction == SCANDirection.DOWN:
            if pos > 0:
                return ElevatorAction.DOWN
            else:
                self.direction = SCANDirection.UP
                return ElevatorAction.IDLE

        return ElevatorAction.IDLE
