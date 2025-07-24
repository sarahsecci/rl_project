import os
import time

import dqn
import gymnasium as gym
import hydra
import rnd_dqn
import torch
import yaml
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


@hydra.main(config_path="../config/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # generate random number for saving results
    rnd = torch.randint(0, 100000, (1,)).item()

    # Set up logging
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(file_path, "../results/models"), exist_ok=True)
    os.makedirs(os.path.join(file_path, "../results/csv"), exist_ok=True)
    os.makedirs(os.path.join(file_path, "../results/config"), exist_ok=True)
    name = None
    if cfg.agent.type == "dqn":
        name = f"{cfg.env.name}_{cfg.agent.type}_seed{cfg.seed}_rnd{rnd}"
    else:
        name = f"{cfg.env.name}_{cfg.agent.type}_{cfg.agent.rnd_type}_seed{cfg.seed}_rnd{rnd}"
    model_path = os.path.join(file_path, "../results/models", f"{name}.pth")
    csv_path = os.path.join(file_path, "../results/csv", f"{name}.csv")
    yaml_path = os.path.join(file_path, "../results/config", f"{name}.yaml")

    # Set gpu as torch device if using RGBImgObsWrapper (and therefore CNN)
    if cfg.env.wrapper == "RGBImgObsWrapper":
        set_gpu()
    print("torch device: ", torch.get_default_device())

    # Build env with wrapper
    env = build_env(cfg.env.name, cfg.env.wrapper)

    # Set seed
    dqn.set_seed(env, cfg.seed)

    # Set up agent
    agent = setup_agent(cfg, env)

    # Start timer
    time_start = time.time()

    # Train agent
    agent.train(cfg.train.num_frames, csv_path, cfg.train.eval_interval)

    # Stop timer
    t = time.time() - time_start
    print(f"Training took {t:.2f} seconds")

    # Save model
    torch.save(
        {"parameters": agent.q.state_dict(), "optimizer": agent.optimizer.state_dict()},
        model_path,
    )

    # Save config to yaml
    config_to_save = OmegaConf.to_container(cfg, resolve=True)
    config_to_save["run_name"] = name  # Add the name to the config
    config_to_save["run_time"] = t
    with open(yaml_path, "w") as f:
        yaml.dump(config_to_save, f)


if __name__ == "__main__":
    main()
