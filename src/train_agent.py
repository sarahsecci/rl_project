import os
import time
from datetime import datetime as dt

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch
from agent.dqn import DQNAgent
from agent.rnd_dqn import RNDDQNAgent
from minigrid.wrappers import FlatObsWrapper, RGBImgObsWrapper
from omegaconf import DictConfig, OmegaConf


class WrongWrapper(Exception):
    pass


def set_gpu():
    """
    Set gpu as compute device (if available)
    """
    device = torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    torch.set_default_device(device)


def build_env(env_name: str, wrapper: str) -> gym.Env:
    """
    Build a gym environment with FlatObsWrapper or RGBImgObsWrapper

    Args:
        env_name (str): Name of the environment to create.
        wrapper (str): The wrapper to apply to the environment.

    Returns:
        gym.Env: The created environment with the specified wrapper.
    """
    env = gym.make(env_name)

    if wrapper == "FlatObsWrapper":
        env = FlatObsWrapper(env)
    elif wrapper == "RGBImgObsWrapper":
        env = RGBImgObsWrapper(env)
    else:
        raise WrongWrapper("Please use FlatObsWrapper or RGBImgObsWrapper")

    return env


def setup_agent(cfg: DictConfig, env: gym.Env, seed: int = None) -> DQNAgent:
    """
    Set up the DQN agent based on the configuration.

    Args:
        cfg (DictConfig): Configuration for the agent.
        env (gym.Env): The environment to use.
        obs_shape (tuple): Shape of the observations.

    Returns:
        DQNAgent: The configured DQN agent.
    """
    agent = None
    agent_kwargs = None

    if seed is None:
        seed = cfg.seed

    if cfg.agent.type == "dqn":
        agent_kwargs = dict(
            buffer_capacity=cfg.agent.buffer_capacity,
            batch_size=cfg.agent.batch_size,
            lr=cfg.agent.dqn_lr,
            gamma=cfg.agent.gamma,
            epsilon_start=cfg.agent.epsilon_start,
            epsilon_final=cfg.agent.epsilon_final,
            epsilon_decay=cfg.agent.epsilon_decay,
            dqn_target_update_freq=cfg.agent.dqn_target_update_freq,
            dqn_hidden_size=cfg.agent.dqn_hidden_size,
            seed=seed,
        )
        agent = DQNAgent(env, **agent_kwargs)
    elif cfg.agent.type == "rnd":
        agent_kwargs = dict(
            buffer_capacity=cfg.agent.buffer_capacity,
            batch_size=cfg.agent.batch_size,
            lr=cfg.agent.dqn_lr,
            gamma=cfg.agent.gamma,
            epsilon_start=cfg.agent.epsilon_start,
            epsilon_final=cfg.agent.epsilon_final,
            epsilon_decay=cfg.agent.epsilon_decay,
            dqn_target_update_freq=cfg.agent.dqn_target_update_freq,
            dqn_hidden_size=cfg.agent.dqn_hidden_size,
            rnd_type=cfg.agent.rnd_type,
            rnd_hidden_size=cfg.agent.rnd_hidden_size,
            rnd_output_size=cfg.agent.rnd_output_size,
            rnd_lr=cfg.agent.rnd_lr,
            rnd_update_freq=cfg.agent.rnd_update_freq,
            rnd_reward_weight=cfg.agent.rnd_reward_weight,
            seed=seed,
        )
        agent = RNDDQNAgent(env, **agent_kwargs)
    else:
        raise ValueError(f"Unknown agent type: {cfg.agent.type}")

    return agent


def set_visitation_map(env_name) -> np.ndarray:
    splits = env_name.split("-")
    env_dim_str = splits[2]
    width, height = map(int, env_dim_str.split("x"))
    visitation_map = np.zeros((height, width), dtype=np.int32)

    return visitation_map


def set_results_dir(cfg: DictConfig, seed: int = None) -> str:
    """
    Set the results directory based on the configuration.

    Returns
    -------
    str
        Path to the results directory
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(file_path, "../results"), exist_ok=True)

    if seed is None:
        seed = cfg.seed

    # Create unique folder name
    if hasattr(cfg, "sweep") and cfg.sweep.run_id is not None:
        if cfg.agent.type == "dqn":
            run_folder = (
                f"sweeps/{cfg.env.name}_{cfg.agent.type}_seed_{seed}/{cfg.sweep.run_id}"
            )
        else:
            run_folder = f"sweeps/{cfg.env.name}_{cfg.agent.type}_{cfg.agent.rnd_type}_seed_{seed}/{cfg.sweep.run_id}"
    else:
        if cfg.agent.type == "dqn":
            run_folder = f"runs/{cfg.env.name}_{cfg.agent.type}_seed_{seed}_time_{dt.now().strftime('%d-%m-%y_%H-%M-%S')}"
        else:
            run_folder = f"runs/{cfg.env.name}_{cfg.agent.type}_{cfg.agent.rnd_type}_seed_{seed}_time_{dt.now().strftime('%d-%m-%y_%H-%M-%S')}"

    run_path = os.path.join(file_path, "../results", run_folder)
    os.makedirs(run_path, exist_ok=True)

    return run_path


def save_results(run_path: str, cfg: DictConfig, agent: DQNAgent, runtime: time):
    # Save model
    model_file = os.path.join(run_path, "model.pth")
    torch.save(
        {"parameters": agent.q.state_dict(), "optimizer": agent.optimizer.state_dict()},
        model_file,
    )

    # Save config
    config_file = os.path.join(run_path, "config.yaml")
    with open(config_file, "w") as f:
        OmegaConf.save(cfg, f)

    # Save training time
    with open(os.path.join(run_path, "runtime.txt"), "w") as f:
        f.write(f"{runtime:.2f}\n")

    print(f"Results saved to: {run_path}")


def train_agent(cfg: DictConfig, seed: int = None) -> str:
    """
    Train an agent and return the results directory path.

    Returns
    -------
    str
        Path to the results directory
    """
    run_path = set_results_dir(cfg, seed)

    env = build_env(cfg.env.name, cfg.env.wrapper)
    agent = setup_agent(cfg, env, seed)

    visitation_map = set_visitation_map(cfg.env.name)

    # Convert num_frames to integer (SMAC passes floats for budget)
    num_frames = int(cfg.train.num_frames)

    # Train agent and measure training time
    time_start = time.time()
    agent.train(
        num_frames=num_frames,
        saving_path=run_path,
        visitation_map=visitation_map,
        vmap_save_every_n=cfg.train.vmap_save_every_n,
        minibatch_window=cfg.train.minibatch_window,
        minibatch_save_every_n=cfg.train.minibatch_save_every_n,
        eval_interval=cfg.train.eval_interval,
        decimals=cfg.train.saved_decimals,
    )
    runtime = time.time() - time_start
    print(f"Training took {runtime:.2f} seconds")

    save_results(run_path, cfg, agent, runtime)

    return run_path


def parse_performance_metric(results_dir: str) -> float:
    """
    Parse the final performance metric from training results.
    Optimizes for minimal episode length (shorter episodes = better performance).

    Parameters
    ----------
    results_dir : str
        Directory containing training results

    Returns
    -------
    float
        Performance metric for optimization (higher is better)
    """
    # Constants
    MAX_EPISODE_STEPS = 360  # Maximum steps before truncation in MiniGrid

    try:
        # Find episode rewards CSV
        episode_csv = os.path.join(results_dir, "episode_rewards.csv")

        if not os.path.exists(episode_csv):
            print(f"Episode rewards file not found: {episode_csv}")
            return -float("inf")

        # Read episode data
        df = pd.read_csv(episode_csv)

        if len(df) < 10:
            print(f"Too few episodes: {len(df)}")
            return -float("inf")

        # Calculate episode lengths from step differences (current_step - previous_step)
        steps = df["steps"].values
        rewards = df["rewards"].values
        episode_lengths = []
        prev_step = 0

        for _, current_step in enumerate(steps):
            episode_length = current_step - prev_step
            episode_lengths.append(episode_length)
            prev_step = current_step

        episode_lengths = np.array(episode_lengths)

        # Get average episode length of last 25% of episodes
        if len(episode_lengths) < 4:
            print("Not enough episodes to calculate average length")
            return -float("inf")
        last_25_percent = int(len(episode_lengths) * 0.75)
        final_episode_lengths = episode_lengths[last_25_percent:]
        avg_episode_length = np.mean(final_episode_lengths)

        # Get amount of successful episodes
        successful_episodes = []

        for i, (length, reward) in enumerate(zip(episode_lengths, rewards)):
            # Consider episode successful if it completed in less than max steps
            if length < MAX_EPISODE_STEPS and reward > 0:
                successful_episodes.append(length)

        # Normalize by max steps
        performance = avg_episode_length / MAX_EPISODE_STEPS

        print(f"Successful episodes: {len(successful_episodes)}")
        print(f"Average episode length: {avg_episode_length:.2f}")
        print(f"Performance metric (normalized length): {performance:.4f}")

        return float(performance)

    except Exception as e:
        print(f"Error parsing performance: {e}")
        return -float("inf")


@hydra.main(config_path="../config/", config_name="rnd_naive_opt", version_base="1.2")
def main(cfg: DictConfig):
    # Set gpu as torch device if using RGBImgObsWrapper (and therefore CNN)
    if cfg.env.wrapper == "RGBImgObsWrapper":
        set_gpu()
    print("torch device: ", torch.get_default_device())

    # Train agent on single seed
    # results_dir = train_agent(cfg)

    # performance = parse_performance_metric(results_dir)

    # # Log trial result
    # if cfg.agent.type == "dqn":
    #     print(f"Run {cfg.agent.type} completed with performance: {performance:.4f}")
    # else:
    #     print(
    #         f"Run {cfg.agent.type}_{cfg.agent.rnd_type} completed with performance: {performance:.4f}"
    #     )

    # Uncomment this to train over multiple seeds
    max_seed = 20  # Set max number of seeds to train over

    for seed in range(max_seed + 1):
        print(f"Training with seed {seed}...")
        results_dir = train_agent(cfg, seed)
        performance = parse_performance_metric(results_dir)
        print(f"Run for seed {seed} completed with performance: {performance:.4f}")


if __name__ == "__main__":
    main()
