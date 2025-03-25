"""look.py: LOOK algorithm agent for elevator control."""

from typing import Any
import numpy as np

from agents.base import BaseAgent
from agents.utils import unpack_flat_observation
from environments.elevator import ElevatorAction
from environments.elevator_environment import ElevatorEnvironmentObservation


class LOOKAgent(BaseAgent):
    """LOOK Algorithm Agent for Elevator Control."""

    def __init__(self, num_floors: int, num_elevators: int):
        super().__init__(num_floors=num_floors, num_elevators=num_elevators)
        assert num_elevators == 1, "LOOK algorithm is designed for a single elevator."
        self.direction = ElevatorAction.UP  # Initial movement direction

    def act(
        self, observation: ElevatorEnvironmentObservation
    ) -> tuple[list[ElevatorAction], dict[str, Any]]:
        if isinstance(observation, np.ndarray):  # Handle flat observations
            observation = unpack_flat_observation(
                observation, num_floors=self.num_floors, num_elevators=self.num_elevators
            )

        current_floor = observation["elevators"]["current_floor"][0]
        requests_up = observation["requests_up"]
        requests_down = observation["requests_down"]

        next_action = self._determine_next_action(
            current_floor=current_floor,
            requests_up=requests_up,
            requests_down=requests_down,
        )
        return [next_action], {}

    def _determine_next_action(
        self, current_floor: int, requests_up: np.ndarray, requests_down: np.ndarray
    ) -> ElevatorAction:
        """Determine the next action based on current position and pending requests."""

        if self.direction == ElevatorAction.UP:
            # Continue moving up if there are requests above
            if np.any(requests_up[current_floor + 1 :]) or np.any(
                requests_down[current_floor + 1 :]
            ):
                return ElevatorAction.UP
            # Reverse direction if no upward requests but there are downward requests
            elif np.any(requests_up[:current_floor]) or np.any(requests_down[:current_floor]):
                self.direction = ElevatorAction.DOWN
                return ElevatorAction.DOWN

        elif self.direction == ElevatorAction.DOWN:
            # Continue moving down if there are requests below
            if np.any(requests_down[:current_floor]) or np.any(requests_up[:current_floor]):
                return ElevatorAction.DOWN
            # Reverse direction if no downward requests but there are upward requests
            elif np.any(requests_up[current_floor + 1 :]) or np.any(
                requests_down[current_floor + 1 :]
            ):
                self.direction = ElevatorAction.UP
                return ElevatorAction.UP

        # Remain idle if no requests are pending
        return ElevatorAction.IDLE
