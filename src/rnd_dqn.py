"""
Deep Q-Learning with RND implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQNAgent
from networks import CNN, MLP


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
        decimals: int = 5,
        rnd_type: str = "naive",  # "naive" or "on_sample"
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
            decimals,
            seed,
        )
        self.seed = seed
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

    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        rnd_error : float
            MSE loss value for the RND update.
        """
        # get first element from each tupel for a list of tupels
        states = [transition[0] for transition in training_batch]
        states = [
            self._process_obs(s) for s in states
        ]  # Process each state individually
        # next_states = np.array([transition[3] for transition in training_batch])

        # Convert states to torch tensors
        # Ensure all states are (C, H, W)
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
        rnd_error.backward()  # computing grandients of rnd_error with backpropagation
        self.rnd_predictor_optimizer.step()  # updating network parameters according to Adam update rule (adjusting weights and minimizing loss)
        return float(rnd_error.item())  # returning scalar value of the loss as float

    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        if isinstance(state, dict):
            state = self._process_obs(state)
        # Ensure state is (C, H, W) if it's an image

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            rnd_predicted = self.rnd_predictor_network(state_tensor)
            rnd_target = self.rnd_target_network(state_tensor)
        rnd_error = torch.mean((rnd_predicted - rnd_target) ** 2)

        return rnd_error.item() * self.rnd_reward_weight

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

        if self.rnd_type == "on_sample":
            # Compute RND bonus on-sample
            rnd_bonus = torch.tensor(
                [self.get_rnd_bonus(s) for s in states], dtype=torch.float32
            )
            r += rnd_bonus

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
        eval_interval : int
            Every this many episodes, print average reward.
        """
        print("Starting training...")
        state, _ = self.env.reset()
        state = self._process_obs(state)
        ep_reward = 0.0
        episode_rewards = []
        steps = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(
                state, evaluate=True
            )  # Use greedy action selection
            next_state, reward, done, truncated, _ = self.env.step(action)
            next_state = self._process_obs(next_state)

            # Apply RND bonus (naive)
            if self.rnd_type == "naive":
                reward += self.get_rnd_bonus(next_state)

            # Store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # Update if buffer is large enough
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

                # Update RND if buffer is large enough
                if self.total_steps % self.rnd_update_freq == 0:
                    self.update_rnd(batch)

            if done or truncated:
                state, _ = self.env.reset()
                state = self._process_obs(state)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0

                # Logging
                if len(episode_rewards) % eval_interval == 0:
                    avg = np.mean(episode_rewards)
                    print(
                        f"Frame {frame}, AvgReward({eval_interval}): {avg:.3f}, ε={self.epsilon():.5f}"
                    )

        print("Training complete.")

        # Save training data to CSV
        training_data = pd.DataFrame(
            {
                "steps": steps,
                "rewards": [round(r, self._decimals) for r in episode_rewards],
            }
        )
        training_data.to_csv(file_path, index=False)
