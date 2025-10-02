from __future__ import annotations
import enum
import numpy as np


class Action(enum.IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


class Square(enum.IntEnum):
    EMPTY = 0
    ROBOT = 1
    OBSTACLE = 2
    ROAD = 3
    STORE = 4


class Gridworld:
    def __init__(
        self,
        N: int,
        robot: np.ndarray = np.empty((2)),
        obstacles: np.ndarray = np.empty((0, 2)),
        roads: np.ndarray = np.empty((0, 2)),
        stores: np.ndarray = np.empty((0, 2)),
    ) -> None:
        """ Gridworld object.

        Args:
            N (int): Determines the size of the map (N x N).
            robot (np.ndarray, optional): The [Y, X] location of the robot. Defaults to np.empty(2).
            obstacles (np.ndarray, optional): Tuples of [Y, X] of obstacles. Defaults to np.empty((0, 2)).
            roads (np.ndarray, optional): Tuples of [Y, X] of roads. Defaults to np.empty((0, 2)).
            stores (np.ndarray, optional): Tuples of [Y, X] of stores. Defaults to np.empty((0, 2)).
        """
        self.N = N
        self.map = np.zeros((N, N))
        self._add_entity(robot[None, ...], Square.ROBOT)
        self._add_entity(obstacles, Square.OBSTACLE)
        self._add_entity(roads, Square.ROAD)
        self._add_entity(stores, Square.STORE)

        self.robot = robot

    def _add_entity(self, locations: np.ndarray, value: Square) -> None:
        self.map[self._to_bool_map(locations)] = value

    def _to_bool_map(self, locations: np.ndarray) -> np.ndarray:
        int_locations = locations.astype(int)
        ret = np.zeros_like(self.map, dtype=bool)
        ret[int_locations[:, 0], int_locations[:, 1]] = True
        return ret

    def is_valid_transition(self, squares: np.ndarray) -> np.ndarray:
        """Returns a np.array which is True for each square tuple which is achievable form current robot position."""
        is_adjacent = np.sum(np.abs(squares - self.robot[None, ...]), axis=1) <= 1
        is_valid = (
            (squares[:, 0] >= 0)
            & (squares[:, 0] < self.N)
            & (squares[:, 1] >= 0)
            & (squares[:, 1] < self.N)
            & (self.map[self._to_bool_map(squares)] != Square.OBSTACLE)
        )
        return is_adjacent & is_valid

    def get_motion_result(self, action: Action, safe_return: bool = True) -> np.ndarray:
        """Get the [Y, X] location result of taking the requested action.

        Args:
            action (Action): action taken.
            safe_return (bool, optional): Applies proper obstacle/edge prevention if true. Defaults to True.

        Returns:
            np.ndarray: [Y, X] location.
        """
        if action == action.UP:
            new_robot = self.robot + np.array([-1, 0])
        elif action == action.DOWN:
            new_robot = self.robot + np.array([1, 0])
        elif action == action.RIGHT:
            new_robot = self.robot + np.array([0, 1])
        elif action == action.LEFT:
            new_robot = self.robot + np.array([0, -1])
        elif action == action.STAY:
            new_robot = self.robot.copy()
        else:
            new_robot = self.robot.copy()
        return np.clip(new_robot, 0, self.N - 1) if safe_return else new_robot


def get_transition_probability(
    a: Action, s: Gridworld, s_prime: Gridworld, p_e: float = 0.2
) -> float:
    """
    Transition probability of (a, s, s')
    """
    # If the next robot state is not possible
    if not s.is_valid_transition(s_prime.robot[None, ...]):
        return 0.0
    # Staying in the same place is an edge case
    if np.all(s_prime.robot == s.robot):
        p = 0.0
        for a_option in Action:
            base_p = 1 - p_e if a == a_option else p_e / 4
            if a_option == Action.STAY or not s.is_valid_transition(
                s.get_motion_result(a_option, safe_return=False)[None, ...]
            ):
                print(a_option.name, base_p)
                p += base_p
        return p
    # If not stay, always in [1 - p_e, p_e / 4, 0]
    if np.all(s_prime.robot == s.get_motion_result(a)):
        return 1 - p_e
    return p_e / 4