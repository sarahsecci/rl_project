import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_episode_rewards(
    csv_files, labels, save_path, plot_epsilon=False, window_size=100
):
    """
    Plots episode rewards from multiple CSV files on the same plot for comparison.

    Args:
        csv_files (list of str): List of paths to CSV files containing episode rewards.
        labels (list of str): List of labels for each line in the plot.
        save_path (str): Path to save the resulting plot.
        plot_epsilon (bool): If True, also plot epsilon on a second y-axis.
        window_size (int): Window size for rolling average of rewards.
    """
    # Sort files and labels by run_id (assuming labels are numeric)
    try:
        sorted_pairs = sorted(zip(csv_files, labels), key=lambda x: int(x[1]))
        csv_files, labels = zip(*sorted_pairs)
    except ValueError:
        # If labels are not numeric, sort alphabetically
        sorted_pairs = sorted(zip(csv_files, labels), key=lambda x: x[1])
        csv_files, labels = zip(*sorted_pairs)

    if plot_epsilon:
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()

        colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))

        for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
            df = pd.read_csv(csv_file)

            # Calculate rolling average for rewards
            rewards_smooth = (
                df["rewards"].rolling(window=window_size, min_periods=1).mean()
            )

            # Plot rewards on left y-axis using steps as x-axis
            ax1.plot(
                df["steps"],
                rewards_smooth,
                label=f"Run {label}",
                linewidth=1.2,
                color=colors[i],
                alpha=0.8,
            )

            # Plot epsilon on right y-axis with dashed line
            ax2.plot(
                df["steps"],
                df["epsilon"],
                linestyle="--",
                alpha=0.4,
                color=colors[i],
                linewidth=0.8,
            )

        ax1.set_xlabel("Steps", fontsize=12)
        ax1.set_ylabel("Average Reward", fontsize=12, color="black")
        ax2.set_ylabel("Epsilon", fontsize=12, color="gray")
        ax1.set_title(
            f"Episode Rewards (Moving Average, window={window_size}) and Epsilon vs Steps",
            fontsize=14,
        )

        # Style the plot
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)

        # Add epsilon label to legend
        ax2.plot([], [], "k--", alpha=0.4, linewidth=0.8, label="Epsilon")
        ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

    else:
        fig, ax = plt.subplots(figsize=(12, 7))

        colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))

        for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
            df = pd.read_csv(csv_file)

            # Calculate rolling average for rewards
            rewards_smooth = (
                df["rewards"].rolling(window=window_size, min_periods=1).mean()
            )

            ax.plot(
                df["steps"],
                rewards_smooth,
                label=f"Run {label}",
                linewidth=1.2,
                color=colors[i],
                alpha=0.8,
            )

        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel("Average Reward", fontsize=12)
        ax.set_title(
            f"Episode Rewards Comparison (Moving Average, window={window_size}) vs Steps",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(file_path, "../../results/sweeps")

    sweep = "MiniGrid-DoorKey-6x6-v0_dqn_seed_0"

    # Check if the sweep directory exists
    sweep_path = os.path.join(results_dir, sweep)
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(f"Sweep directory does not exist: {sweep_path}")

    # Find episode_rewards.csv files in subdirectories of sweep_path
    episode_rewards_files = []
    for subdir in os.listdir(sweep_path):
        subdir_path = os.path.join(sweep_path, subdir)
        rewards_file = os.path.join(subdir_path, "episode_rewards.csv")
        if os.path.isdir(subdir_path) and os.path.isfile(rewards_file):
            episode_rewards_files.append(rewards_file)
    if not episode_rewards_files:
        raise FileNotFoundError(
            f"No episode_rewards.csv files found in subdirectories of: {sweep_path}"
        )

    # Extract labels from folder names
    labels = [os.path.basename(os.path.dirname(f)) for f in episode_rewards_files]

    # Plot rewards only with moving average
    save_path_rewards = os.path.join(sweep_path, "episode_rewards_comparison.png")
    plot_episode_rewards(
        episode_rewards_files, labels, save_path_rewards, plot_epsilon=False
    )
    print(f"Rewards-only plot saved to: {save_path_rewards}")

    # Plot rewards with epsilon and moving average
    save_path_with_epsilon = os.path.join(
        sweep_path, "episode_rewards_with_epsilon_comparison.png"
    )
    plot_episode_rewards(
        episode_rewards_files, labels, save_path_with_epsilon, plot_epsilon=True
    )
    print(f"Rewards with epsilon plot saved to: {save_path_with_epsilon}")
