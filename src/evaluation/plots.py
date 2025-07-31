import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve


def _plot_single_seed(df: pd.DataFrame, x_col: str, y_col: str):
    """Plot data for a single seed."""
    plt.plot(df[x_col], df[y_col])


def _plot_multiple_seeds(file_paths: list[str], x_col: str, y_col: str, ylabel: str):
    """Plot aggregated data for multiple seeds using rliable."""
    n_seeds = len(file_paths)

    # Read data from different runs and add seed column
    dfs = []
    min_length = float("inf")

    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)
        df["seed"] = i
        dfs.append(df)
        min_length = min(min_length, len(df))

    # Truncate all dataframes to the same length
    for i in range(len(dfs)):
        dfs[i] = dfs[i].head(min_length)

    # Combine the dataframes
    df = pd.concat(dfs, ignore_index=True)

    # Make sure only one set of steps is attempted to be plotted
    x_values = df[x_col].to_numpy().reshape((n_seeds, -1))[0]

    # Reshape data for rliable
    train_scores = {"rnd_naive": df[y_col].to_numpy().reshape((n_seeds, -1))}

    # Aggregate using IQM (Interquartile Mean)
    iqm = lambda scores: np.array(
        [
            metrics.aggregate_iqm(scores[:, eval_idx])
            for eval_idx in range(scores.shape[-1])
        ]
    )

    iqm_scores, iqm_cis = get_interval_estimates(
        train_scores,
        iqm,
        reps=2000,
    )

    # Plot using rliable's sample efficiency curve
    plot_sample_efficiency_curve(
        x_values,
        iqm_scores,
        iqm_cis,
        algorithms=["rnd_naive"],
        xlabel=x_col.title(),
        ylabel=f"IQM {ylabel}",
        marker="",
        linewidth=1,
    )


def plot_metric_over_steps(
    file_paths: list[str],
    save_path: str,
    metric_col: str = "rewards",
    title: str = None,
    x_col: str = "steps",
    ylabel: str = None,
):
    """
    Generic function to plot any metric over steps for one or multiple seeds.

    Args:
        file_paths: List of CSV file paths
        save_path: Path to save the plot
        metric_col: Column name for the metric to plot (e.g., 'reward', 'td_error')
        title: Plot title (if None, auto-generated)
        x_col: Column name for x-axis (default: 'step')
        ylabel: Y-axis label (if None, uses metric_col)
    """
    # Set defaults
    if title is None:
        title = f"{metric_col.replace('_', ' ').title()} over {x_col.title()}"
    if ylabel is None:
        ylabel = metric_col.replace("_", " ").title()

    # Create new figure
    plt.figure(figsize=(10, 6))

    if len(file_paths) == 1:
        df = pd.read_csv(file_paths[0])
        _plot_single_seed(df, x_col, metric_col)
    else:
        _plot_multiple_seeds(file_paths, x_col, metric_col, ylabel)

    plt.title(title)
    plt.xlabel(x_col.title())
    plt.ylabel(ylabel)
    plt.legend()
    # plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)


def rewards_over_steps(file_paths: list[str], save_path: str):
    """Plot rewards over steps. Wrapper for plot_metric_over_steps."""
    plot_metric_over_steps(
        file_paths=file_paths,
        save_path=save_path,
        metric_col="rewards",
        title="Reward over Steps",
        ylabel="Reward",
    )


def td_over_steps(file_paths: list[str], save_path: str):
    """Plot TD errors over steps. Wrapper for plot_metric_over_steps."""
    plot_metric_over_steps(
        file_paths=file_paths,
        save_path=save_path,
        metric_col="td_error",
        title="TD Error over Steps",
        ylabel="TD Error",
    )


def rewards_over_episodes(file_paths: list[str], save_path: str):
    """Plot rewards over episodes. Wrapper for plot_metric_over_steps."""
    plot_metric_over_steps(
        file_paths=file_paths,
        save_path=save_path,
        metric_col="reward",
        x_col="episode",
        title="Reward over Episodes",
        ylabel="Reward",
    )


def minibatch_rewards_over_steps(file_paths: list[str], save_path: str):
    """Plot minibatch rewards over steps. Wrapper for plot_metric_over_steps."""
    plot_metric_over_steps(
        file_paths=file_paths,
        save_path=save_path,
        metric_col="minibatch_reward",
        title="Minibatch Reward over Steps",
        ylabel="Minibatch Reward",
    )


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(file_path, "../../results/runs")

    ep_rewards = [
        os.path.join(
            results_dir,
            "MiniGrid-DoorKey-6x6-v0_rnd_naive_seed_0_time_31-07-25_09-03-20/episode_rewards.csv",
        ),
        os.path.join(
            results_dir,
            "MiniGrid-DoorKey-6x6-v0_rnd_naive_seed_1_time_31-07-25_09-24-30/episode_rewards.csv",
        ),
    ]
    mini_rewards = [
        os.path.join(
            results_dir,
            "MiniGrid-DoorKey-6x6-v0_rnd_naive_seed_0_time_31-07-25_09-03-20/minibatch_rewards.csv",
        ),
        os.path.join(
            results_dir,
            "MiniGrid-DoorKey-6x6-v0_rnd_naive_seed_1_time_31-07-25_09-24-30/minibatch_rewards.csv",
        ),
    ]

    rewards_steps_path = os.path.join(
        file_path, "../../results/runs/plots/rnd_naive_rewards_over_steps.png"
    )
    mkdir = os.makedirs(os.path.dirname(rewards_steps_path), exist_ok=True)
    rewards_over_steps(ep_rewards, rewards_steps_path)
    # rewards_over_episodes(ep_rewards, os.path.join(file_path, "../results/runs/dqn/rewards_over_episodes_seed_0_1.png"))
    # minibatch_rewards_over_steps(mini_rewards, os.path.join(file_path, "../results/runs/dqn/minibatch_rewards_over_steps_seed_0_1.png"))
    # td_over_steps(mini_rewards, os.path.join(file_path, "../results/runs/dqn/td_over_steps_seed_0_1.png"))
