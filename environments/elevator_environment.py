"""elevator_environment.py: Elevator environment."""

import time
from typing import TypedDict

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Dict, MultiBinary, MultiDiscrete, Box

from environments.elevator import Elevator, ElevatorAction
from environments.workload_scenario import (
    RandomPassengerWorkloadScenario,
    WorkloadScenario,
)


class ElevatorsObs(TypedDict):
    """Observation of the elevators."""

    current_floor: np.ndarray  # shape: (num_elevators,)
    current_load: np.ndarray  # shape: (num_elevators,)


class ElevatorEnvironmentObservation(TypedDict):
    """Observation of the elevator environment."""

    elevators: ElevatorsObs
    requests_up: np.ndarray  # shape: (num_floors,)
    requests_down: np.ndarray  # shape: (num_floors,)


class ElevatorEnvironment(Env):
    """The elevator environment."""

    FLOOR_UP_TRAVEL_PENALTY = 0.12
    FLOOR_DOWN_TRAVEL_PENALTY = 0.08
    FLOOR_IDLE_PENALTY = 0.01
    FLOOR_DOOR_PENALTY = 0.05
    UNPRODUCTIVE_STOP_PENALTY = 0.1
    SUCCESSFUL_LOAD_REWARD = 1
    SUCCESSFUL_UNLOAD_REWARD = 2
    WAIT_PENALTY_PER_STEP = 0.03
    TRAVEL_PENALTY_PER_STEP = 0.02

    def __init__(
        self,
        num_elevators: int = 1,
        num_floors: int = 10,
        workload_scenario: WorkloadScenario = None,
        elevator_capacities: list[int] | int = 10,
        min_length: int = 25,
        max_length: int = 1000,
        delay: float = 1.0,
        seed: int = None,
    ):
        self.num_elevators = num_elevators
        self.num_floors = num_floors
        if workload_scenario is None:
            workload_scenario = RandomPassengerWorkloadScenario(
                num_floors,
                spawn_prob=0.2,
                start_floor_probs=[1 / num_floors] * num_floors,
                end_floor_probs=[1 / num_floors] * num_floors,
            )
        self.workload_scenario = workload_scenario
        if isinstance(elevator_capacities, int):
            elevator_capacities = [elevator_capacities] * num_elevators
        elif isinstance(elevator_capacities, list):
            assert (
                len(elevator_capacities) == num_elevators
            ), "Number of elevators and elevator capacities do not match"
        else:
            raise ValueError("Invalid elevator capacities")
        self.elevator_capacities = elevator_capacities
        self.delay = delay
        self.min_length = min_length
        self.max_length = max_length

        # create elevators
        self.elevators = []
        for i in range(num_elevators):
            elevator = Elevator(num_floors, capacity=elevator_capacities[i])
            elevator.reset()
            self.elevators.append(elevator)

        # create hidden state variables (passenger requests)
        self.passenger_requests = []

        # create action and observation spaces
        self.action_space = MultiDiscrete([len(ElevatorAction.action_space())] * num_elevators)
        self.observation_space = self._get_observation_space()

        self.step_count = 0
        self.served_requests = 0

        self.reset(seed=seed)

    @property
    def total_requests(self) -> int:
        """Total number of requests."""
        return len(self.passenger_requests) + self.served_requests

    def _get_requests_up_down(self):
        requests_up = np.zeros(self.num_floors)
        requests_down = np.zeros(self.num_floors)
        for request in self.passenger_requests:
            # only consider requests that are not assigned to an elevator
            if request.current_elevator_index is None:
                if request.start_floor < request.end_floor:
                    requests_up[request.start_floor] += request.num_passengers
                else:
                    requests_down[request.start_floor] += request.num_passengers
        return requests_up, requests_down

    def _get_observation_space(self):
        return Dict(
            {
                "elevators": Dict(
                    {
                        "current_floor": MultiDiscrete([self.num_floors] * self.num_elevators),
                        "current_load": Box(
                            0, max(self.elevator_capacities), shape=(self.num_elevators,)
                        ),
                        "internal_requests": MultiBinary([self.num_floors] * self.num_elevators),
                    }
                ),
                "requests_up": MultiBinary(self.num_floors),
                "requests_down": MultiBinary(self.num_floors),
            }
        )

    def _get_observation(self):
        elevators = {
            "current_floor": np.array([e.current_floor for e in self.elevators]),
            "current_load": np.array([e.current_load for e in self.elevators]),
            "internal_requests": np.array([e.internal_requests for e in self.elevators]),
        }

        requests_up, requests_down = self._get_requests_up_down()

        return {
            "elevators": elevators,
            "requests_up": (requests_up != 0),
            "requests_down": (requests_down != 0),
        }

    def reset(self, seed: int = None, options: dict = None):
        """Reset the environment."""
        self.step_count = 0
        for elevator in self.elevators:
            elevator.reset()
        self.passenger_requests = []
        self.served_requests = 0
        self.elevator_times = [0.0] * self.num_elevators
        if seed is None:
            seed = np.random.randint(0, 1000)
        np.random.seed(seed)
        return self._get_observation(), {}

    def step(self, actions: list[ElevatorAction]):
        """Perform an action on the environment."""

        reward = 0.0
        env_infos = {"served_requests": []}

        # Apply wait and travel penalties per step
        for request in self.passenger_requests:
            if request.current_elevator_index is None:
                reward -= self.WAIT_PENALTY_PER_STEP
            else:
                reward -= self.TRAVEL_PENALTY_PER_STEP

        for elevator_idx, (elevator, action) in enumerate(zip(self.elevators, actions)):

            # update elevator state (excluding load/unload)
            elevator.apply_action(action)

            # update reward based on movement cost
            reward -= {
                ElevatorAction.UP: self.FLOOR_UP_TRAVEL_PENALTY,
                ElevatorAction.DOWN: self.FLOOR_DOWN_TRAVEL_PENALTY,
                ElevatorAction.IDLE: self.FLOOR_IDLE_PENALTY,
                ElevatorAction.STOP: self.FLOOR_DOOR_PENALTY,
            }.get(action, 0.0)

            # if action is STOP also update elevator load/unload
            if action == ElevatorAction.STOP:

                unproductive_stop = True

                for request in sorted(
                    self.passenger_requests,
                    key=lambda r: r.current_elevator_index is None,  # prioritize assigned requests
                ):
                    if request.current_elevator_index is None:  # request not yet served
                        if request.start_floor == elevator.current_floor:

                            # carry out request load
                            if elevator.current_load + request.num_passengers <= elevator.capacity:
                                elevator.current_load += request.num_passengers
                                elevator.internal_requests[request.end_floor] = True
                                request.current_elevator_index = elevator_idx
                                request.load_step = self.step_count
                                reward += self.SUCCESSFUL_LOAD_REWARD * request.num_passengers
                                unproductive_stop = False

                    else:  # request is being served
                        if request.current_elevator_index == elevator_idx:
                            if request.end_floor == elevator.current_floor:
                                # carry out request unload
                                elevator.current_load -= request.num_passengers
                                request.current_elevator_index = None
                                request.unload_step = self.step_count
                                reward += self.SUCCESSFUL_UNLOAD_REWARD * request.num_passengers
                                env_infos["served_requests"].append(request)
                                self.passenger_requests.remove(request)
                                self.served_requests += 1
                                unproductive_stop = False

                if unproductive_stop:
                    reward -= self.UNPRODUCTIVE_STOP_PENALTY

        # workload scenario appends new requests
        self.passenger_requests += self.workload_scenario.step(self.step_count)

        observation = self._get_observation()
        done = self._check_done()

        if not done:
            self.step_count += 1
            time.sleep(self.delay)

        return observation, reward, done, False, env_infos

    def render(self):
        pass

    def _check_done(self):
        return self.step_count >= self.max_length
