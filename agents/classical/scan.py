from enum import Enum
import numpy as np

from agents.base import BaseAgent
from environments.elevator import ElevatorAction
from environments.elevator_environment import ElevatorEnvironmentObservation
from agents.utils import unpack_flat_observation


class SCANDirection(Enum):
    """Direction of the SCAN algorithm."""

    UP = 1
    DOWN = -1


class SCANAgent(BaseAgent):
    """SCAN (Elevator) Agent. Moves in one direction until end is reached, then reverses."""

    def __init__(self, num_floors: int, num_elevators: int):
        super().__init__(num_elevators=num_elevators, num_floors=num_floors)
        assert num_elevators == 1, "SCAN works clearly only with 1 elevator."
        self.direction = ElevatorAction.UP  # Start moving UP initially

    def act(self, observation: ElevatorEnvironmentObservation) -> list[ElevatorAction]:
        if isinstance(observation, np.ndarray):  # SB3-style flat observation
            observation = unpack_flat_observation(
                observation, num_floors=self.num_floors, num_elevators=self.num_elevators
            )

        elevator_position = observation["elevators"]["current_floor"][0]
        requests_up = observation["requests_up"]
        requests_down = observation["requests_down"]

        next_action = self._get_next_action(
            pos=elevator_position,
            requests_up=requests_up,
            requests_down=requests_down,
        )
        return [next_action]

    def _get_next_action(self, pos: int, requests_up: np.ndarray, requests_down: np.ndarray):
        """Explicit logic for correct SCAN movement and direction reversal."""

        if self.direction == ElevatorAction.UP:
            # if there are requests above current position, keep going UP
            if np.any(requests_up[pos + 1 :]) or np.any(requests_down[pos + 1 :]):
                return ElevatorAction.UP

            # If no requests upwards, but there are below, switch DOWN
            if np.any(requests_up[:pos]) or np.any(requests_down[:pos]):
                self.direction = ElevatorAction.DOWN
                return ElevatorAction.DOWN

        elif self.direction == ElevatorAction.DOWN:
            # if there are requests below, continue downward
            if np.any(requests_down[:pos]) or np.any(requests_up[:pos]):
                return ElevatorAction.DOWN

            # no more downward requests, reverse to upward
            if np.any(requests_up[pos + 1 :]) or np.any(requests_down[pos + 1 :]):
                self.direction = ElevatorAction.UP
                return ElevatorAction.UP

        # if no requests anywhere, explicitly go IDLE
        return ElevatorAction.IDLE
