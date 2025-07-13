import os
import time

import dqn
import gymnasium as gym
import hydra
import rnd_dqn
import torch
from minigrid.wrappers import RGBImgObsWrapper
from omegaconf import DictConfig

# Set the default device (gpu if available) for PyTorch
device = torch.device(
    f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
)
torch.set_default_device(device)
print(torch.get_default_device())


def build_env(env_name: str) -> gym.Env:
    """
    Build a gym environment with RGB image observations.

    Args:
        env_name (str): Name of the environment to create.

    Returns:
        gym.Env: The created environment with RGB image observations.
    """
    env = gym.make(env_name)
    env = RGBImgObsWrapper(env)  # Use RGB images for DQN
    obs, _ = env.reset()
    obs_shape = obs["image"].shape
    obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W) for PyTorch
    return env, obs_shape


def setup_agent(cfg: DictConfig, env: gym.Env, obs_shape: tuple) -> dqn.DQNAgent:
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
            seed=cfg.seed,
        )
        agent = dqn.DQNAgent(env, obs_shape, **agent_kwargs)
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
            seed=cfg.seed,
        )
        agent = rnd_dqn.RNDDQNAgent(env, obs_shape, **agent_kwargs)
    else:
        raise ValueError(f"Unknown agent type: {cfg.agent.type}")

    return agent


@hydra.main(config_path="../config/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # Set up logging
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(file_path, "../results/models"), exist_ok=True)
    os.makedirs(os.path.join(file_path, "../results/csv"), exist_ok=True)
    name = None
    if cfg.agent.type == "dqn":
        name = f"{cfg.env.name}_{cfg.agent.type}_seed{cfg.seed}_frames{cfg.train.num_frames}"
    else:
        name = f"{cfg.env.name}_{cfg.agent.type}_{cfg.agent.rnd_type}_seed{cfg.seed}_frames{cfg.train.num_frames}"
    model_path = os.path.join(file_path, "../results/models", f"{name}.pth")
    csv_path = os.path.join(file_path, "../results/csv", f"{name}.csv")

    # Build env with wrapper
    env, obs_shape = build_env(cfg.env.name)

    # Set seed
    dqn.set_seed(env, cfg.seed)

    # Set up agent
    agent = setup_agent(cfg, env, obs_shape)

    # Start timer
    time_start = time.time()

    # Train agent
    agent.train(cfg.train.num_frames, csv_path, cfg.train.eval_interval)

    # Stop timer
    print(f"Training took {time.time() - time_start:.2f} seconds")

    # Save model
    torch.save(
        {"parameters": agent.q.state_dict(), "optimizer": agent.optimizer.state_dict()},
        model_path,
    )


if __name__ == "__main__":
    main()
