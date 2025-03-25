"""scan.py: SCAN elevator agent."""

from enum import Enum
import numpy as np
from typing import Any
from agents.base import BaseAgent
from environments.elevator import ElevatorAction
from environments.elevator_environment import ElevatorEnvironmentObservation
from agents.utils import unpack_flat_observation


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
        """
        if isinstance(observation, np.ndarray):
            observation = unpack_flat_observation(
                observation, num_floors=self.num_floors, num_elevators=self.num_elevators
            )

        elevator_position = observation["elevators"]["current_floor"][0]
        next_action = self._get_next_action(elevator_position)
        return [next_action], {}

    def _get_next_action(self, pos: int) -> ElevatorAction:
        """Determine the next action based on the current position.

        Args:
            pos: The current position of the elevator.

        Returns:
            The next action to take
        """
        if self.direction == SCANDirection.UP:
            if pos < self.num_floors - 1:
                return ElevatorAction.UP
            self.direction = SCANDirection.DOWN
            return ElevatorAction.DOWN

        if self.direction == SCANDirection.DOWN:
            if pos > 0:
                return ElevatorAction.DOWN
            self.direction = SCANDirection.UP
            return ElevatorAction.UP

        return ElevatorAction.IDLE
