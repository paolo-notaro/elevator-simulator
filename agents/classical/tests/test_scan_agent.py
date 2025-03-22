"""test_scan_agent.py: Test cases for the SCAN elevator agent."""

import pytest
import numpy as np
from agents.classical.scan import SCANAgent
from environments.elevator import ElevatorAction
from environments.elevator_environment import ElevatorEnvironmentObservation


@pytest.fixture
def scan_agent():
    """Return a SCAN agent."""
    return SCANAgent(num_floors=10, num_elevators=1)


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


def test_initial_direction(scan_agent: SCANAgent):
    """Test the initial direction of the SCAN agent."""
    assert scan_agent.direction == ElevatorAction.UP


def test_act_no_requests(scan_agent: SCANAgent, observation: ElevatorEnvironmentObservation):
    """Test the SCAN agent when there are no requests."""
    actions = scan_agent.act(observation)
    assert actions == [ElevatorAction.IDLE]


def test_act_requests_up(scan_agent: SCANAgent, observation: ElevatorEnvironmentObservation):
    """Test the SCAN agent when there are requests to go up."""
    observation["requests_up"][5] = True
    actions = scan_agent.act(observation)
    assert actions == [ElevatorAction.UP]


def test_act_requests_down(scan_agent: SCANAgent, observation: ElevatorEnvironmentObservation):
    scan_agent.direction = ElevatorAction.DOWN
    observation["requests_down"][2] = True
    observation["elevators"]["current_floor"][0] = 5
    actions = scan_agent.act(observation)
    assert actions == [ElevatorAction.DOWN]


def test_direction_reversal(scan_agent: SCANAgent, observation: ElevatorEnvironmentObservation):
    observation["requests_up"][9] = True
    scan_agent.act(observation)
    observation["elevators"]["current_floor"][0] = 9
    observation["requests_up"][9] = False
    observation["requests_down"][0] = True
    actions = scan_agent.act(observation)
    assert scan_agent.direction == ElevatorAction.DOWN
    assert actions == [ElevatorAction.DOWN]
