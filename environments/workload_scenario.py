from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PassengerRequest:
    """A passenger request."""

    start_floor: int
    end_floor: int
    num_passengers: int
    current_elevator_index: int = None
    creation_step: int = None
    creation_time: float = None
    load_time: float = None
    unload_time: float = None

    @property
    def wait_time(self):
        """The time between the request being created and the passengers being loaded."""
        if self.load_time is None or self.creation_time is None:
            return None
        return self.load_time - self.creation_time

    @property
    def travel_time(self):
        """The time between the passengers being loaded and unloaded."""
        if self.load_time is None or self.unload_time is None:
            return None
        return self.unload_time - self.load_time


class WorkloadScenario(ABC):
    """A workload scenario for generating passenger requests."""

    def __init__(self, num_floors: int):
        self.num_floors = num_floors

    @abstractmethod
    def step(self, step_count: int) -> list[PassengerRequest]:
        """Generate the next step in the workload scenario.

        Args:
            step_count: The current step count.

        Returns:
            A list of passenger requests

        """
        raise NotImplementedError


class RandomPassengerWorkloadScenario(WorkloadScenario):
    """A workload scenario where a single passenger request is spawned randomly with
    fixed probability."""

    def __init__(
        self,
        num_floors: int,
        spawn_prob: float,
        start_floor_probs: float | list[float],
        end_floor_probs: float | list[float],
    ):
        super().__init__(num_floors)
        self.spawn_prob = spawn_prob
        if isinstance(start_floor_probs, float):
            start_floor_probs = [start_floor_probs] * num_floors
        if isinstance(end_floor_probs, float):
            end_floor_probs = [end_floor_probs] * num_floors
        self.start_floor_probs = start_floor_probs
        self.end_floor_probs = end_floor_probs

    def step(self, step_count: int) -> list[PassengerRequest]:
        """Generate the next step in the workload scenario.

        Args:
            step_count: The current step count.

        Returns:
            A list of passenger requests
        """
        if np.random.rand() < self.spawn_prob:
            start_floor = np.random.choice(self.num_floors, p=self.start_floor_probs)
            end_floor = np.random.choice(self.num_floors, p=self.end_floor_probs)
            while start_floor == end_floor:
                end_floor = np.random.choice(self.num_floors, p=self.end_floor_probs)
            request = PassengerRequest(
                start_floor=start_floor,
                end_floor=end_floor,
                num_passengers=1,
                creation_step=step_count,
            )
            return [request]

        return []
