from typing import Any, Dict, List, Tuple

import numpy as np
from agent import AbstractBuffer


class ReplayBuffer(AbstractBuffer):
    """
    Simple FIFO replay buffer.

    Stores tuples of (state, action, reward, next_state, done, info),
    and evicts the oldest when capacity is exceeded.
    """

    def __init__(self, capacity: int) -> None:
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

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
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
        return [
            (
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.dones[i],
                self.infos[i],
            )
            for i in idxs
        ]

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self.states)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Proportional Prioritized Replay Buffer.

    Extends ReplayBuffer by storing per‐transition priorities and sampling
    with P(i) ∝ priority[i]^α. Also computes importance‐sampling weights
    w(i) ∝ (1/N * 1/P(i))^β to correct bias.
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        """
        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store.
        alpha : float
            How much prioritization is used (0 = uniform, 1 = full prioritization).
        """
        super().__init__(capacity)
        assert alpha >= 0, "Alpha must be non‐negative"
        self.alpha = alpha
        # mirror the buffer length with a list of floats
        self.priorities: List[float] = []

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """
        Add a transition with max priority so it gets sampled at least once.

        New transitions are given priority = current max priority (or 1 if empty).
        """
        # determine new priority
        max_prio = max(self.priorities) if self.priorities else 1.0
        # add to base buffers
        super().add(state, action, reward, next_state, done, info)
        # pop oldest priority if over capacity
        if len(self.priorities) >= self.capacity:
            self.priorities.pop(0)
        self.priorities.append(max_prio)

    def sample(
        self, batch_size: int = 32, beta: float = 0.4
    ) -> Tuple[List[Tuple[Any, Any, float, Any, bool, Dict]], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions according to their priorities.

        Parameters
        ----------
        batch_size : int
        beta : float
            Importance‐sampling exponent (0 = no correction, 1 = full correction).

        Returns
        -------
        transitions : list of (s, a, r, s', done, info)
        indices      : np.ndarray of sampled indices (for priority updates)
        weights      : np.ndarray of importance‐sampling weights, shape (batch_size,)
        """
        N = len(self.priorities)
        if N == 0:
            raise ValueError("Cannot sample from an empty buffer")

        # compute sampling probabilities
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios**self.alpha
        probs /= probs.sum()

        # draw indices
        indices = np.random.choice(N, batch_size, p=probs, replace=False)
        # get transitions
        transitions = [
            (
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.dones[i],
                self.infos[i],
            )
            for i in indices
        ]

        # compute importance‐sampling weights
        # w_i = (N * P(i))^(-β) / max_j w_j
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        return transitions, indices, weights

    def update_priorities(
        self, indices: List[int], new_priorities: List[float]
    ) -> None:
        """
        After learning from a batch, update the priorities of those sampled.

        Parameters
        ----------
        indices : list of int
            Positions in the buffer to update.
        new_priorities : list of float
            Corresponding new priority values (e.g. abs(td_error) + ε).
        """
        for idx, prio in zip(indices, new_priorities):
            assert prio >= 0, "Priority must be non‐negative"
            self.priorities[idx] = prio
