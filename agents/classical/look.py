"""look.py: LOOK algorithm agent for elevator control."""

from typing import Any
import numpy as np

from agents.base import BaseAgent
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

        current_floor = observation["elevators"]["current_floor"][0]
        internal_requests = observation["elevators"]["internal_requests"][0]
        requests_up = observation["requests_up"]
        requests_down = observation["requests_down"]

        next_action = self._determine_next_action(
            current_floor=current_floor,
            internal_requests=internal_requests,
            requests_up=requests_up,
            requests_down=requests_down,
        )
        return [next_action], {}

    def _determine_next_action(
        self,
        current_floor: int,
        internal_requests: np.ndarray,
        requests_up: np.ndarray,
        requests_down: np.ndarray,
    ) -> ElevatorAction:
        """Determine the next action based on current position and pending requests."""

        # Stop if someone inside needs to get off
        if internal_requests[current_floor]:
            return ElevatorAction.STOP

        # Stop if someone wants to get on in the same direction
        if self.direction == ElevatorAction.UP and requests_up[current_floor]:
            return ElevatorAction.STOP
        elif self.direction == ElevatorAction.DOWN and requests_down[current_floor]:
            return ElevatorAction.STOP

        # check if we should continue in the same direction or change
        if self.direction == ElevatorAction.UP:
            if (
                np.any(internal_requests[current_floor + 1 :])
                or np.any(requests_up[current_floor + 1 :])
                or np.any(requests_down[current_floor + 1 :])
            ):
                return ElevatorAction.UP
            else:
                self.direction = ElevatorAction.DOWN
                # Check for stop again in new direction
                if requests_down[current_floor]:
                    return ElevatorAction.STOP
                return ElevatorAction.DOWN

        elif self.direction == ElevatorAction.DOWN:
            if (
                np.any(internal_requests[:current_floor])
                or np.any(requests_down[:current_floor])
                or np.any(requests_up[:current_floor])
            ):
                return ElevatorAction.DOWN
            else:
                self.direction = ElevatorAction.UP
                if requests_up[current_floor]:
                    return ElevatorAction.STOP
                return ElevatorAction.UP

        return ElevatorAction.IDLE
