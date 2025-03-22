"""base.py: Base class for all agents"""

from abc import ABC, abstractmethod

from environments.elevator import ElevatorAction
from environments.elevator_environment import ElevatorEnvironmentObservation


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, num_floors, num_elevators):
        self.num_floors = num_floors
        self.num_elevators = num_elevators

    @abstractmethod
    def act(self, observation: ElevatorEnvironmentObservation) -> list[ElevatorAction]:
        """Returns the actions to take given the current observation."""
        raise NotImplementedError("act method not implemented")

    def __str__(self) -> str:
        """Return the class name as a string."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return the class name as a string."""
        return self.__class__.__name__
