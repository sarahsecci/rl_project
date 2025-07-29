import os
import time
from datetime import datetime as dt

import dqn
import gymnasium as gym
import hydra
import numpy as np
import rnd_dqn
import torch
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


def setup_agent(cfg: DictConfig, env: gym.Env) -> dqn.DQNAgent:
    """
    Set up the DQN agent based on the configuration.

    Args:
        cfg (DictConfig): Configuration for the agent.
        env (gym.Env): The environment to use.
        obs_shape (tuple): Shape of the observations.

    Returns:
        dqn.DQNAgent: The configured DQN agent.
    """
    agent = None
    agent_kwargs = None

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
            decimals=cfg.train.saved_decimals,
            seed=cfg.seed,
        )
        agent = dqn.DQNAgent(env, **agent_kwargs)
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
            decimals=cfg.train.saved_decimals,
            rnd_type=cfg.agent.rnd_type,
            rnd_hidden_size=cfg.agent.rnd_hidden_size,
            rnd_output_size=cfg.agent.rnd_output_size,
            rnd_lr=cfg.agent.rnd_lr,
            rnd_update_freq=cfg.agent.rnd_update_freq,
            rnd_reward_weight=cfg.agent.rnd_reward_weight,
            seed=cfg.seed,
        )
        agent = rnd_dqn.RNDDQNAgent(env, **agent_kwargs)
    else:
        raise ValueError(f"Unknown agent type: {cfg.agent.type}")

    return agent


def set_results_dir(cfg: DictConfig) -> str:
    """
    Set the results directory based on the configuration.

    Returns
    -------
    str
        Path to the results directory
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(file_path, "../results"), exist_ok=True)

    # Create unique folder name
    if hasattr(cfg, "sweep") and cfg.sweep.run_id is not None:
        if cfg.agent.type == "dqn":
            run_folder = f"sweeps/{cfg.env.name}_{cfg.agent.type}_seed_{cfg.seed}/{cfg.sweep.run_id}"
        else:
            run_folder = f"sweeps/{cfg.env.name}_{cfg.agent.type}_{cfg.agent.rnd_type}_seed_{cfg.seed}/{cfg.sweep.run_id}"
    else:
        run_folder = f"runs/{cfg.env.name}_{cfg.agent.type}_seed_{cfg.seed}_time_{dt.now().isoformat()}"

    run_path = os.path.join(file_path, "../results", run_folder)

    # Check if sweep run_id folder exists
    # if hasattr(cfg, 'sweep') and cfg.sweep.run_id is not None and os.path.exists(run_path):
    #     raise FileExistsError(f"Results directory for sweep run_id '{cfg.sweep.run_id}' already exists: {run_path}")

    os.makedirs(run_path, exist_ok=True)

    return run_path


def train_agent(cfg: DictConfig) -> str:
    """
    Train an agent and return the results directory path.

    Returns
    -------
    str
        Path to the results directory
    """
    # Get env name and agent type
    env_name = cfg.env.name
    agent_type = cfg.agent.type
    if agent_type == "rnd":  # Override agent type if RND agent
        rnd_type = cfg.agent.rnd_type
        agent_type = f"{agent_type}_{rnd_type}"

    # Set results directory and model/config file paths
    run_path = set_results_dir(cfg)
    model_file = os.path.join(run_path, "model.pth")
    config_file = os.path.join(run_path, "config.yaml")

    # Build env with wrapper
    env = build_env(env_name, cfg.env.wrapper)

    # get env dimension
    splits = env_name.split("-")
    env_dim_str = splits[2]
    width, height = map(int, env_dim_str.split("x"))
    visitation_map = np.zeros((height, width), dtype=np.int32)

    # Set up agent
    agent = setup_agent(cfg, env)

    # Start timer
    time_start = time.time()

    # Train agent
    agent.train(cfg.train.num_frames, run_path, visitation_map, cfg.train.eval_interval)

    # Stop timer
    t = time.time() - time_start
    print(f"Training took {t:.2f} seconds")

    # Save model
    torch.save(
        {"parameters": agent.q.state_dict(), "optimizer": agent.optimizer.state_dict()},
        model_file,
    )

    # Save config and training time
    with open(config_file, "w") as f:
        OmegaConf.save(cfg, f)
    with open(os.path.join(run_path, "runtime.txt"), "w") as f:
        f.write(f"{t:.2f}\n")

    print(f"Results saved to: {run_path}")

    return run_path  # Return the path for sweep analysis


# def visu_trained_agent():


@hydra.main(config_path="../config/", config_name="dqn", version_base="1.2")
def main(cfg: DictConfig):
    # Set gpu as torch device if using RGBImgObsWrapper (and therefore CNN)
    if cfg.env.wrapper == "RGBImgObsWrapper":
        set_gpu()
    print("torch device: ", torch.get_default_device())

    # train agent
    train_agent(cfg)

    # visualize trained agent


if __name__ == "__main__":
    main()
