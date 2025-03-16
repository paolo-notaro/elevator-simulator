"""elevator.py: A simple elevator environment."""

from enum import Enum


class ElevatorAction(Enum):
    """The actions that the elevator can take."""

    UP = 0
    DOWN = 1
    IDLE = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.name}"

    @classmethod
    def action_space(cls):
        """Return the action space."""
        return list(cls)


class Elevator:
    """The elevator environment."""

    doors_open: bool
    blocked: bool
    current_floor: int

    def __init__(
        self,
        num_floors: int,
        capacity: int,
    ):
        self.floors = num_floors
        self.capacity = capacity
        self.reset()

    def reset(self) -> None:
        """Reset the elevator to its initial state."""
        self.current_floor = 0
        self.current_load = 0
        self.doors_open = False
        self.blocked = False
        self.internal_requests = [False] * self.floors

    def apply_action(self, action: ElevatorAction) -> None:
        """Perform an action on the elevator.

        Args:
            action: The action to perform.

        """
        # Perform the action on the elevator
        if action == ElevatorAction.UP:
            if self.current_floor < self.floors - 1:
                self.current_floor += 1
                self.internal_requests[self.current_floor] = False

        elif action == ElevatorAction.DOWN:
            if self.current_floor > 0:
                self.current_floor -= 1
                self.internal_requests[self.current_floor] = False
        elif action == ElevatorAction.IDLE:
            pass
        else:
            raise ValueError("Invalid action")
