"""
Visualize a trained DQN agent in MiniGrid environment.
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


def visualize_agent(cfg: DictConfig, num_episodes=3, max_steps=360):
    env_name = cfg.env.name
    file_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        file_path,
        "../../results/runs/MiniGrid-DoorKey-6x6-v0_rnd_naive_seed_1_time_31-07-25_09-24-30/model.pth",
    )

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
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=2)
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

        print(f"Episode {ep + 1}: Reward={ep_reward}, Steps={step}")

        # Display animation using matplotlib
        fig = plt.figure()
        plt.axis("off")
        ims = [[plt.imshow(f, animated=True)] for f in frames]
        ani = animation.ArtistAnimation(
            fig, ims, interval=300, blit=True, repeat_delay=1000
        )
        ani_path = os.path.join(
            file_path, f"../../results/runs/test/trained_agent_{ep + 1}.gif"
        )
        ani.save(ani_path, writer="pillow")


# change config name
@hydra.main(
    config_path="../../results/runs/MiniGrid-DoorKey-6x6-v0_rnd_naive_seed_1_time_31-07-25_09-24-30",
    config_name="config.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    visualize_agent(cfg, num_episodes=1)


if __name__ == "__main__":
    main()
