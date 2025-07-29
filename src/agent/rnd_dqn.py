"""
Deep Q-Learning with RND (naive and on-sample) implementation.
"""

from typing import Any, Dict, List, Tuple

import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from agent.dqn import DQNAgent
from agent.networks import CNN, MLP
from agent.replay_buffer import ReplayBuffer


class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

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
        rnd_type: str = "naive",
        rnd_hidden_size: int = 64,
        rnd_output_size: int = 64,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_reward_weight: float = 0.1,
        seed: int = 0,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
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
            How many updates between target-network syncs.
        dqn_hidden_size : int
            Hidden layer size for DQN.
        rnd_type : str
            Type of RND ("naive" or "on_sample").
        rnd_hidden_size : int
            Hidden layer size for RND networks.
        rnd_output_size : int
            Output size for RND networks.
        rnd_lr : float
            Learning rate for RND networks.
        rnd_update_freq : int
            How often to update RND networks.
        rnd_reward_weight : float
            Weight for RND bonus in reward.
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
        self.buffer = ReplayBuffer(buffer_capacity, intr=True)

        self.rnd_lr = rnd_lr
        self.rnd_update_freq = rnd_update_freq
        self.rnd_reward_weight = rnd_reward_weight

        # Initialize RND networks
        if isinstance(env.observation_space, gym.spaces.Box):
            self.rnd_predictor_network = MLP(
                self.obs_shape, rnd_output_size, rnd_hidden_size
            )
            self.rnd_target_network = MLP(
                self.obs_shape, rnd_output_size, rnd_hidden_size
            )
        elif isinstance(env.observation_space, gym.spaces.Dict):
            self.rnd_predictor_network = CNN(
                self.obs_shape, rnd_output_size, rnd_hidden_size
            )
            self.rnd_target_network = CNN(
                self.obs_shape, rnd_output_size, rnd_hidden_size
            )

        # Do not redefine self.optimizer here; it is already set in DQNAgent
        self.rnd_predictor_optimizer = optim.Adam(
            self.rnd_predictor_network.parameters(), lr=self.rnd_lr
        )
        self.rnd_type = rnd_type

    def _update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
            states are already processed by _process_obs.

        Returns
        -------
        rnd_error : float
            MSE loss value for the RND update.
        """
        # Get first element from each tupel for a list of tupels
        states = [transition[0] for transition in training_batch]

        # Convert states to torch tensors
        states = [
            np.transpose(s, (2, 0, 1))
            if (isinstance(s, np.ndarray) and s.ndim == 3 and s.shape[-1] == 3)
            else s
            for s in states
        ]
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)

        # Compute MSE
        rnd_predicted = self.rnd_predictor_network(states_tensor)
        rnd_target = self.rnd_target_network(states_tensor).detach()
        rnd_error = torch.nn.MSELoss()(rnd_predicted, rnd_target)

        # Update the RND predictor network
        self.rnd_predictor_optimizer.zero_grad()  # resetting gradients of all model parameters to zero
        rnd_error.backward()  # Computing grandients of rnd_error with backpropagation
        self.rnd_predictor_optimizer.step()  # Updating network parameters according to Adam update rule (adjusting weights and minimizing loss)

        return float(rnd_error.item())  # Returning scalar value of the loss as float

    def _get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment. Already processed by _process_obs.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            rnd_predicted = self.rnd_predictor_network(state_tensor)
            rnd_target = self.rnd_target_network(state_tensor)
        rnd_error = torch.mean((rnd_predicted - rnd_target) ** 2)

        return rnd_error.item() * self.rnd_reward_weight

    def _update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict, float]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, extrinsic_reward, next_state, done, info, intrinsic_reward).
            state and next_state are already processed by _process_obs.

        Returns
        -------
        mean_extr : float
            Mean extrinsic reward for minibatch
        mean_intr : float
            Mean intrinsic reward for minibatch
        loss : float
            MSE loss value = TD error
        td_std : float
            Standard deviation of TD error for minibatch
        """
        # Unpack
        states, actions, extr_rewards, next_states, dones, _, intr_rewards = zip(
            *training_batch
        )

        s = torch.tensor(np.array(states), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        extr_r = torch.tensor(np.array(extr_rewards), dtype=torch.float32)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)
        mask = torch.tensor(np.array(dones), dtype=torch.float32)
        intr_r = torch.tensor(np.array(intr_rewards), dtype=torch.float32)

        # Compute RND bonus on-sample
        if self.rnd_type == "on_sample":
            intr_r = torch.tensor(
                [self._get_rnd_bonus(s) for s in states], dtype=torch.float32
            )

        # Calculate mean extrinsic and intrinsic reward in the sampled minibatch
        mean_extr = extr_r.mean().item()
        mean_intr = intr_r.mean().item()

        # Get combined reward for each sample
        r = extr_r + intr_r

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
            float(mean_intr),
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
            - For minibatch values: list of tuples (frame, extrinsic, intrinsic, loss, td_std)
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
                elif len(first) == 5:
                    # Minibatch values: (frame, extrinsic, intrinsic, loss, td_std)
                    df = pd.DataFrame(
                        training_data,
                        columns=["steps", "extrinsic", "intrinsic", "loss", "td_std"],
                    )
                    df[["extrinsic", "intrinsic", "loss", "td_std"]] = df[
                        ["extrinsic", "intrinsic", "loss", "td_std"]
                    ].round(decimals)
                    # Calculate rolling mean first, then subsample
                    df_means = (
                        df[["extrinsic", "intrinsic", "loss", "td_std"]]
                        .rolling(window=mean_window)
                        .mean()
                    )
                    df_means["steps"] = df["steps"]
                    df_means = df_means.dropna()
                    df_means = df_means[
                        ["steps", "extrinsic", "intrinsic", "loss", "td_std"]
                    ]
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
            next_state, extr_reward, done, truncated, _ = self.env.step(action)
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

            # Apply RND bonus (naive)
            intr_reward = 0.0
            if self.rnd_type == "naive":
                intr_reward = self._get_rnd_bonus(next_state)

            # Store transition in replay buffer
            self.buffer.add(
                state,
                action,
                extr_reward,
                next_state,
                done or truncated,
                {},
                intr_reward,
            )
            state = next_state
            ep_reward += extr_reward + intr_reward

            # Update agent if buffer is large enough to sample minibatch
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                extr, intr, loss, td_std = self._update_agent(batch)
                minibatch_values.append((frame, extr, intr, loss, td_std))

                # Update RND if buffer is large enough
                if self.total_steps % self.rnd_update_freq == 0:
                    self._update_rnd(batch)

            # Reset environment if episode is done or truncated and save last episodes reward
            if done or truncated:
                state, _ = self.env.reset(seed=self.seed)
                state = self._process_obs(state)
                episode_rewards.append((frame, ep_reward, self._epsilon()))
                ep_reward = 0.0

                # Logging mean episode reward every eval_interval episodes (for terminal)
                if len(episode_rewards) % eval_interval == 0:
                    avg = np.mean(episode_rewards[-eval_interval:])
                    print(
                        f"Frame {frame}, AvgReward({eval_interval}): {avg:.3f}, ε={self._epsilon():.5f}"
                    )

        print("Training complete.")

        # Save training data using the unified method
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
