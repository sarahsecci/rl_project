"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from agent import AbstractAgent
from buffers import ReplayBuffer
from networks import CNN


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # Some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q‐Learning agent with ε‐greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        obs_shape: tuple[int, ...],
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 5000,
        dqn_target_update_freq: int = 1000,
        dqn_hidden_size: int = 64,
        seed: int = 0,
    ) -> None:
        """
        Initialize replay buffer, Q‐networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini‐batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        dqn_target_update_freq : int
            How many updates between target‐network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            dqn_target_update_freq,
            dqn_hidden_size,
            seed,
        )
        self.env = env
        set_seed(env, seed)

        # Handle different observation space types
        if obs_shape is not None:
            # Use provided obs_dim (for MiniGrid and other complex spaces)
            self.obs_dim = obs_shape[0] * obs_shape[1] * obs_shape[2]
            self._obs_key = None
        elif isinstance(env.observation_space, gym.spaces.Box):
            # Simple Box space (like CartPole, LunarLander)
            self.obs_dim = env.observation_space.shape[0]
            self._obs_key = None
        elif isinstance(env.observation_space, gym.spaces.Dict):
            # Dict space (like MiniGrid)
            self._obs_key = "image"
            self.obs_shape = env.observation_space[self._obs_key].shape
            self.obs_dim = int(np.prod(self.obs_shape))
        else:
            raise NotImplementedError(
                f"Unsupported observation space type: {type(env.observation_space)}"
            )

        n_actions = env.action_space.n

        self.q = CNN(obs_shape, n_actions, dqn_hidden_size)
        self.target_q = CNN(obs_shape, n_actions, dqn_hidden_size)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # Hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.dqn_target_update_freq = dqn_target_update_freq

        self.total_steps = 0  # For ε decay and target sync

    def _process_obs(self, obs):
        """
        Process observation based on observation space type.

        Parameters
        ----------
        obs : Any
            Raw observation from the environment.
        """
        # Handle dict observations (like MiniGrid)
        if isinstance(obs, dict):
            if "image" in obs:
                obs = obs["image"]
            else:
                # If no "image" key, take the first available key
                obs = list(obs.values())[0]

        # Handle the case where _obs_key is set
        if hasattr(self, "_obs_key") and self._obs_key is not None:
            if isinstance(obs, dict):
                obs = obs[self._obs_key]

        return np.asarray(obs, dtype=np.float32)

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -1.0 * self.total_steps / self.epsilon_decay
        )

    def predict_action(
        self, state: np.ndarray, info: Dict[str, Any] = {}, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε‐greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        info_out : dict
            Empty dict (compatible with interface).
        """
        state = self._process_obs(state)

        if evaluate:
            # Purely greedy
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qvals = self.q(t)
            action = int(torch.argmax(qvals, dim=1).item())
        else:
            # ε-greedy
            if np.random.rand() < self.epsilon():
                action = self.env.action_space.sample()
            else:
                t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    qvals = self.q(t)
                action = int(torch.argmax(qvals, dim=1).item())

        return action

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # Unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)

        # Process observations
        states = [self._process_obs(s) for s in states]
        next_states = [self._process_obs(s) for s in next_states]

        s = torch.tensor(np.array(states), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(np.array(rewards), dtype=torch.float32)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)
        mask = torch.tensor(np.array(dones), dtype=torch.float32)

        # Current Q estimates for taken actions
        pred = self.q(s).gather(1, a).squeeze(1)

        # Compute TD target with frozen network
        with torch.no_grad():
            next_q = self.target_q(s_next).max(1)[0]
            target = r + self.gamma * next_q * (1 - mask)

        loss = nn.MSELoss()(pred, target)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Occasionally sync target network
        if self.total_steps % self.dqn_target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(self, num_frames: int, file_path: str, eval_interval: int = 10) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        file_path : str
            Path to save training data (CSV).
        eval_interval : int
            Every this many episodes, print average reward.
        """
        print("Starting training...")
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        steps = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # Store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # Update if buffer is large enough
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0

                # Logging
                if len(recent_rewards) % eval_interval == 0:
                    avg = np.mean(recent_rewards[-eval_interval:])
                    print(
                        f"Frame {frame}, AvgReward({eval_interval}): {avg:.3f}, ε={self.epsilon():.5f}"
                    )

        print("Training complete.")

        # Save training data to CSV
        training_data = pd.DataFrame({"steps": steps, "rewards": episode_rewards})
        training_data.to_csv(file_path, index=False)
