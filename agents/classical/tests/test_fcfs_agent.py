"""test_fcfs_agent.py: Test cases for the FCFS elevator agent."""

import pytest
import numpy as np
from agents.classical.fcfs import FCFSAgent
from environments.elevator import ElevatorAction
from environments.elevator_environment import ElevatorEnvironmentObservation
from copy import deepcopy


@pytest.fixture
def fcfs_agent():
    """Return a FCFS agent."""
    return FCFSAgent(num_floors=10, num_elevators=1)


@pytest.fixture
def observation():
    """Return an observation."""
    requests_up = np.zeros(10, dtype=bool)
    requests_down = np.zeros(10, dtype=bool)
    return {
        "elevators": {
            "current_floor": np.array([0]),
            "current_load": np.array([0]),
        },
        "requests_up": requests_up,
        "requests_down": requests_down,
    }


def test_initial_state(fcfs_agent: FCFSAgent):
    """Test the initial state of the FCFS agent."""
    observation = {
        "elevators": {
            "current_floor": np.array([0]),
            "current_load": np.array([0]),
        },
        "requests_up": np.zeros(10),
        "requests_down": np.zeros(10),
    }
    actions = fcfs_agent.act(observation)
    assert actions == [ElevatorAction.IDLE]


def test_single_request_up(fcfs_agent: FCFSAgent, observation: ElevatorEnvironmentObservation):
    """Test the FCFS agent with a single request to go up."""
    fcfs_agent.last_observation = deepcopy(observation)
    observation["requests_up"][5] = True
    actions = fcfs_agent.act(observation)
    assert actions == [ElevatorAction.UP]


def test_single_request_down(fcfs_agent: FCFSAgent, observation: ElevatorEnvironmentObservation):
    """Test the FCFS agent with a single request to go down."""
    fcfs_agent.last_observation = deepcopy(observation)
    observation["requests_down"][2] = True
    observation["elevators"]["current_floor"][0] = 5
    actions = fcfs_agent.act(observation)
    assert actions == [ElevatorAction.DOWN]


def test_multiple_requests(fcfs_agent: FCFSAgent, observation: ElevatorEnvironmentObservation):
    """Test the FCFS agent with multiple requests."""
    fcfs_agent.last_observation = deepcopy(observation)
    observation["requests_up"][5] = True
    observation["requests_down"][2] = True
    actions = fcfs_agent.act(observation)
    assert actions == [ElevatorAction.UP]
    fcfs_agent.act(observation)  # Move to the next step
    actions = fcfs_agent.act(observation)
    assert actions == [ElevatorAction.UP]
