"""
Visualize a trained DQN agent in MiniGrid environment.
"""

import os

import gymnasium as gym
import matplotlib
import torch
from minigrid.wrappers import RGBImgObsWrapper

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import rnd_dqn


def visualize_agent(model_path, env_name, num_episodes=3, max_steps=100):
    # Build environment and use RGB Wrapper
    env = gym.make(env_name, render_mode="rgb_array")
    env = RGBImgObsWrapper(env)  # Use RGB images for DQN
    obs, _ = env.reset()
    obs_shape = obs["image"].shape
    obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W) for PyTorch

    # Instantiate agent and load weights
    agent = rnd_dqn.RNDDQNAgent(
        env=env,
        obs_shape=obs_shape,
        dqn_hidden_size=32,
        rnd_hidden_size=32,
        rnd_output_size=32,
        rnd_type="on_sample",
        rnd_reward_weight=0.01,
    )
    checkpoint = torch.load(model_path, map_location="cpu")
    agent.q.load_state_dict(checkpoint["parameters"])
    agent.q.eval()

    # Run n episodes
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        step = 0
        ep_reward = 0.0
        frames = []

        while not (done or truncated) and step < max_steps:
            frame = env.render()  # Get RGB array
            frames.append(frame)
            action = agent.predict_action(obs, evaluate=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            step += 1

        print(f"Episode {ep + 1}: Reward={ep_reward}, Steps={step}")

        # Display animation using matplotlib
        fig = plt.figure()
        plt.axis("off")
        ims = [[plt.imshow(f, animated=True)] for f in frames]
        ani = animation.ArtistAnimation(
            fig, ims, interval=100, blit=True, repeat_delay=1000
        )
        ani.save(f"results/agent_run_{ep + 1}.gif", writer="pillow")


if __name__ == "__main__":
    # Edit the model path and environment name as needed
    file_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        file_path,
        "../../results/models/MiniGrid-DoorKey-6x6-v0_rnd_on_sample_seed2_frames100000.pth",
    )
    env_name = "MiniGrid-DoorKey-6x6-v0"

    # Visualize the agent
    visualize_agent(model_path, env_name, num_episodes=3)
