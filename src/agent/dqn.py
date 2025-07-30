"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from abstract import AbstractAgent
from agent.networks import CNN, MLP
from agent.replay_buffer import ReplayBuffer


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
        self.seed = seed
        self.env.reset(seed=self.seed)

        n_actions = env.action_space.n

        # Handle Flat Observation Space with MLP and RGB Observation Space with CNN
        if isinstance(env.observation_space, gym.spaces.Box):
            obs = env.observation_space.shape
            self.obs_shape = obs[0]
            self.q = MLP(self.obs_shape, n_actions, dqn_hidden_size)
            self.target_q = MLP(self.obs_shape, n_actions, dqn_hidden_size)
        elif isinstance(env.observation_space, gym.spaces.Dict):
            obs = env.observation_space
            obs_shape = obs["image"].shape
            self.obs_shape = (
                obs_shape[2],
                obs_shape[0],
                obs_shape[1],
            )  # (C, H, W) for PyTorch
            self.obs_dim = obs_shape[0] * obs_shape[1] * obs_shape[2]
            self._obs_key = None
            self.q = CNN(self.obs_shape, n_actions, dqn_hidden_size)
            self.target_q = CNN(self.obs_shape, n_actions, dqn_hidden_size)

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

    def _epsilon(self) -> float:
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

    def _predict_action(
        self, state: np.ndarray, info: Dict[str, Any] = {}, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε‐greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation. Alreadsy processed by _process_obs.
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
        if evaluate:
            # Purely greedy
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qvals = self.q(t)
            action = int(torch.argmax(qvals, dim=1).item())
        else:
            # ε-greedy
            if np.random.rand() < self._epsilon():
                action = self.env.action_space.sample()
            else:
                t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    qvals = self.q(t)
                action = int(torch.argmax(qvals, dim=1).item())

        return action

    def _update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> tuple[float, float, float]:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, extrinsic_reward, next_state, done, info).
            state and next_state are already processed by _process_obs.

        Returns
        -------
        mean_extr : float
            Mean extrinsic reward for minibatch
        loss : float
            MSE loss value = TD error
        td_std : float
            Standard deviation of TD error for minibatch
        """
        # Unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)

        s = torch.tensor(np.array(states), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(np.array(rewards), dtype=torch.float32)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)
        mask = torch.tensor(np.array(dones), dtype=torch.float32)

        # Calculate mean extrinsic reward in the sampled minibatch
        mean_extr = r.mean().item()

        # Current Q estimates for taken actions
        pred = self.q(s).gather(1, a).squeeze(1)

        # Compute TD target with frozen network
        with torch.no_grad():
            next_q = self.target_q(s_next).max(1)[0]
            target = r + self.gamma * next_q * (1 - mask)

        # Compute loss = TD error
        loss = nn.MSELoss()(pred, target)

        # Compute standard deviation of TD error
        td_std = (pred - target).std().item()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Occasionally sync target network
        if self.total_steps % self.dqn_target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1

        return [
            float(mean_extr),
            float(loss.item()),
            float(td_std),
        ]

    def _save_training_data(
        self,
        saving_path: str,
        file_name: str,
        training_data,
        mean_window: int = 100,
        save_every_n: int = 20,
        decimals: int = 5,
    ) -> None:
        """
        Save training data to CSV files.
        Handles visitation map (np.ndarray), episode rewards (list of tuples),
        and minibatch values (list of tuples).

        Parameters
        ----------
        saving_path : str
            Path to save the CSV file.
        file_name : str
            Name of the CSV file.
        training_data : np.ndarray or list of tuples
            Data to save:
            - For visitation map: np.ndarray
            - For episode rewards: list of tuples (frame, reward, epsilon)
            - For minibatch values: list of tuples (frame, extrinsic, loss, td_std)
        mean_window : int, optional
            For minibatch data: rolling window size for averaging
        save_every_n : int
            For minibatch data: save only every nth data point to reduce file size
        """
        os.makedirs(saving_path, exist_ok=True)
        file_path = os.path.join(saving_path, file_name)

        if isinstance(training_data, np.ndarray):
            # Visitation map
            df = pd.DataFrame(training_data)
            df.to_csv(file_path, index=False)
        elif isinstance(training_data, list) and len(training_data) > 0:
            first = training_data[0]
            if isinstance(first, tuple):
                if len(first) == 3:
                    # Episode rewards: (frame, reward, epsilon)
                    df = pd.DataFrame(
                        training_data, columns=["steps", "rewards", "epsilon"]
                    )
                    df["rewards"] = df["rewards"].round(decimals)
                    df["epsilon"] = df["epsilon"].round(decimals)
                    df.to_csv(file_path, index=False)
                elif len(first) == 4:
                    # Minibatch values: (frame, extrinsic, loss, td_std)
                    df = pd.DataFrame(
                        training_data, columns=["steps", "extrinsic", "loss", "td_std"]
                    )
                    df[["extrinsic", "loss", "td_std"]] = df[
                        ["extrinsic", "loss", "td_std"]
                    ].round(decimals)
                    # Calculate rolling mean first, then subsample
                    df_means = (
                        df[["extrinsic", "loss", "td_std"]]
                        .rolling(window=mean_window)
                        .mean()
                    )
                    df_means["steps"] = df["steps"]
                    df_means = df_means.dropna()
                    df_means = df_means[["steps", "extrinsic", "loss", "td_std"]]
                    # Subsample the averaged data
                    df_final = df_means.iloc[::save_every_n]
                    df_final.to_csv(file_path, index=False)
                else:
                    raise ValueError("Unknown tuple structure in training_data list.")
            else:
                raise ValueError(
                    "List elements must be tuples for episode/minibatch data."
                )
        else:
            raise ValueError("Unsupported training_data type for saving.")

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

    def train(
        self,
        num_frames: int,
        saving_path: str,
        visitation_map: np.ndarray,
        vmap_save_every_n: int = 50000,
        minibatch_window: int = 100,
        minibatch_save_every_n: int = 20,
        eval_interval: int = 10,
        decimals: int = 5,
    ) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        saving_path : str
            Path to save training data (CSV).
        visitation_map : np.ndarray
            Visitation map to track agent position.
        vmap_save_every_n : int, optional
            Save visitation map every nth step to reduce file size.
        minibatch_window : int, optional
            Rolling window size for averaging minibatch values.
        minibatch_save_every_n : int, optional
            Save minibatch values every nth step to reduce file size.
        eval_interval : int
            Print average episode reward every eval_interval steps in terminal.
        """
        print("Starting training...")
        state, _ = self.env.reset(seed=self.seed)
        state = self._process_obs(state)
        ep_reward = 0.0
        episode_rewards = []
        minibatch_values = []

        for frame in range(1, num_frames + 1):
            action = self._predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            next_state = self._process_obs(next_state)

            # Log agent position for visitation map
            base_env = self.env.unwrapped
            x, y = base_env.agent_pos
            visitation_map[y, x] += 1

            # Save visitation map every vmap_save_every_n steps
            if frame % vmap_save_every_n == 0:
                vmap_file = os.path.join(saving_path, f"visitation_map_{frame}.csv")
                self._save_training_data(
                    saving_path, vmap_file, visitation_map, decimals=decimals
                )

            # Store transition in replay buffer
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # Update agent if buffer is large enough to sample minibatch
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                extr, loss, td_std = self._update_agent(batch)
                minibatch_values.append((frame, extr, loss, td_std))

            # Reset environment if episode is done or truncated and save last episodes reward
            if done or truncated:
                state, _ = self.env.reset(seed=self.seed)
                state = self._process_obs(state)
                episode_rewards.append((frame, ep_reward, self._epsilon()))
                ep_reward = 0.0

                # Logging mean episode reward every eval_interval episodes (for terminal)
                if len(episode_rewards) % eval_interval == 0:
                    recent_rewards = [ep[1] for ep in episode_rewards[-eval_interval:]]
                    avg = np.mean(recent_rewards)
                    print(
                        f"Frame {frame}, AvgReward({eval_interval}): {avg:.3f}, ε={self._epsilon():.5f}"
                    )

        print("Training complete.")

        # Save training data to CSV files
        self._save_training_data(
            saving_path,
            f"visitation_map_{num_frames}.csv",
            visitation_map,
            decimals=decimals,
        )
        self._save_training_data(
            saving_path, "episode_rewards.csv", episode_rewards, decimals=decimals
        )
        self._save_training_data(
            saving_path,
            "minibatch_values.csv",
            minibatch_values,
            mean_window=minibatch_window,
            save_every_n=minibatch_save_every_n,
            decimals=decimals,
        )
