"""
Visualization script for trained DQN and RND-DQN agents in MiniGrid environments.
Authors: Clara Schindler and Sarah Secci
Date: 09-08-25
Parts of this code were made with the help of Copilot
"""

import os

import gymnasium as gym
import hydra
import matplotlib
import torch
from agent.dqn import DQNAgent
from agent.rnd_dqn import RNDDQNAgent
from minigrid.wrappers import FlatObsWrapper
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def visualize_agent(
    cfg: DictConfig,
    file_path: str,
    num_seeds: int = 3,
    start_seed: int = 0,
    max_steps: int = 360,
):
    """
    Visualize a trained agent's behavior in the environment and save as GIF animations.

    Parameters
    ----------
    cfg : DictConfig
        Configuration used for training the agent.
    file_path : str
        Path to the directory containing the trained model.
    num_seeds : int
        Number of episodes to run with different seeds.
    start_seed : int
        Starting seed for episode generation.
    max_steps : int
        Maximum steps per episode before truncation.
    """
    env_name = cfg.env.name
    model_path = os.path.join(file_path, "model.pth")

    # Build environment and use RGB Wrapper
    env = gym.make(env_name, render_mode="rgb_array")
    env = FlatObsWrapper(env)
    env.reset()

    # Instantiate agent and load weights
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
            seed=cfg.seed,
        )
        agent = RNDDQNAgent(env, **agent_kwargs)
    else:
        raise ValueError(f"Unknown agent type: {cfg.agent.type}")

    # Load model parameters
    checkpoint = torch.load(model_path, map_location="cpu")
    agent.q.load_state_dict(checkpoint["parameters"])
    agent.q.eval()

    # Run n episodes
    for s in range(num_seeds):
        obs, _ = env.reset(seed=s + start_seed)
        done = False
        truncated = False
        step = 0
        ep_reward = 0.0
        frames = []

        while not (done or truncated) and step < max_steps:
            frame = env.render()  # Get RGB array
            frames.append(frame)
            action = agent._predict_action(obs, evaluate=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            step += 1

        print(f"Episode {s}: Reward={ep_reward}, Steps={step}")

        # Display animation using matplotlib
        fig = plt.figure()
        plt.axis("off")
        ims = [[plt.imshow(f, animated=True)] for f in frames]
        ani = animation.ArtistAnimation(
            fig, ims, interval=300, blit=True, repeat_delay=1000
        )

        ani_path = os.path.join(file_path, f"trained_agent_{s + start_seed}.gif")
        ani.save(ani_path, writer="pillow")


# change config name
@hydra.main(
    config_path="../../results/final_runs/MiniGrid-DoorKey-6x6-v0_dqn_seed_0_time_02-08-25_22-16-03",
    config_name="config.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    """
    Main function for visualizing a trained agent.

    Parameters
    ----------
    cfg : DictConfig
        Configuration loaded from the trained model directory.
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        current_path,
        "../../results/final_runs/MiniGrid-DoorKey-6x6-v0_dqn_seed_0_time_02-08-25_22-16-03",
    )

    visualize_agent(cfg, file_path, num_seeds=1, start_seed=0)


if __name__ == "__main__":
    main()
