"""elevator.py: A simple elevator environment."""

from enum import Enum


class ElevatorAction(Enum):
    """The actions that the elevator can take."""

    UP = 0
    DOWN = 1
    IDLE = 2
    STOP = 3

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

    IDLE_TIME = 1.0  # time to wait when idle
    FLOOR_TRAVEL_TIME = 1.0  # time to travel one floor
    DOOR_OPEN_TIME = 2.0  # time to open doors
    DOOR_CLOSE_TIME = 2.0  # time to close doors

    doors_open: bool
    current_floor: int
    current_load: int
    internal_requests: list[bool]
    internal_time: float = 0.0
    steps_to_time: dict[int, float] = dict()

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
        self.internal_requests = [False] * self.floors
        self.internal_time = 0.0
        self.steps_to_time = {}

    def apply_action(self, action: ElevatorAction, step_count: int) -> None:
        """Perform an action on the elevator.

        Args:
            action: The action to perform.

        """

        # Perform the action on the elevator
        if action == ElevatorAction.UP:
            if self.current_floor < self.floors - 1:
                self.doors_open = False
                self.current_floor += 1
                self.internal_time += self.FLOOR_TRAVEL_TIME

        elif action == ElevatorAction.DOWN:
            if self.current_floor > 0:
                self.doors_open = False
                self.current_floor -= 1
                self.internal_time += self.FLOOR_TRAVEL_TIME
        elif action == ElevatorAction.IDLE:
            self.doors_open = False
            self.internal_time += self.IDLE_TIME
        elif action == ElevatorAction.STOP:
            self.internal_time += self.DOOR_OPEN_TIME
            self.doors_open = True
            self.internal_requests[self.current_floor] = False
            self.internal_time += self.DOOR_CLOSE_TIME
        else:
            raise ValueError("Invalid action")

        self.steps_to_time[step_count] = self.internal_time  # memorize the time
