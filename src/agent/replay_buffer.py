from typing import Any, Dict, List, Tuple

import numpy as np
from abstract import AbstractBuffer


class ReplayBuffer(AbstractBuffer):
    """
    Simple FIFO replay buffer.

    Stores tuples of (state, action, reward, next_state, done, info),
    and evicts the oldest when capacity is exceeded.
    """

    def __init__(self, capacity: int, intr: bool = False) -> None:
        """
        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store.
        """
        super().__init__()
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.infos: List[Dict] = []

        self.intr = intr
        if self.intr:
            self.intr_rewards: List[float] = []

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
        intr_reward: float = None,
    ) -> None:
        """
        Add a single transition to the buffer.

        If the buffer is full, the oldest transition is removed.

        Parameters
        ----------
        state : np.ndarray
            Observation before action.
        action : int or float
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Observation after action.
        done : bool
            Whether episode terminated/truncated.
        info : dict
            Gym info dict (can store extras).
        intr_reward : float, optional
            Intrinsic reward (if applicable, e.g. in RND).
        """
        if len(self.states) >= self.capacity:
            # pop oldest
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.infos.pop(0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.infos.append(info)

        if self.intr:
            if len(self.intr_rewards) >= self.capacity:
                self.intr_rewards.pop(0)
            self.intr_rewards.append(intr_reward)

    def sample(
        self, batch_size: int = 32
    ) -> List[Tuple[Any, Any, float, Any, bool, Dict]]:
        """
        Uniformly sample a batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        List of transitions as (state, action, reward, next_state, done, info).
        """
        idxs = np.random.choice(len(self.states), batch_size, replace=False)

        result = [
            (
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.dones[i],
                self.infos[i],
                self.intr_rewards[i] if self.intr else None,
            )
            for i in idxs
        ]
        if self.intr:
            return result
        else:
            # Remove the last element (intr_reward) from each tuple
            return [t[:-1] for t in result]

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self.states)
