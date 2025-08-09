"""
Plot generation for comparing DQN, RND-naive, and RND-on-sample agents using rliable library.
Authors: Clara Schindler and Sarah Secci
Date: 09-08-25
Parts of this code were made with the help of Copilot
"""

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve
from scipy import interpolate


def load_and_align_data(
    file_paths: list[str],
    x_col: str,
    y_col: str,
    step_interval: int = 1000,
    min_step: int = 0,
    max_step: int = 60000,
):
    """
    Load CSV files of different lengths and align them to a common step grid.

    Parameters
    ----------
    file_paths : list[str]
        List of paths to CSV files (one per seed).
    x_col : str
        Name of the x-axis column (e.g., "steps").
    y_col : str
        Name of the y-axis column (e.g., "rewards").
    step_interval : int
        Interval for common step grid.
    min_step : int
        Minimum step value for interpolation.
    max_step : int
        Maximum step value for interpolation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        common_steps: Array of common step values
        aligned_data: 2D array of shape (n_seeds, n_steps)
    """
    # Load all data and find step range
    dfs = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        dfs.append(df)

    if not dfs:
        return np.array([]), np.array([])

    # Create common step grid
    common_steps = np.arange(min_step, max_step + step_interval, step_interval)

    # Interpolate each seed to common grid
    aligned_data = []
    for i, df in enumerate(dfs):
        f = interpolate.interp1d(
            df[x_col].values,
            df[y_col].values,
            kind="linear",
            bounds_error=False,
            fill_value=(
                df[y_col].iloc[0],
                df[y_col].iloc[-1],
            ),  # take the first and last value to interpolate from min to max step
        )
        interpolated_values = f(common_steps)
        aligned_data.append(interpolated_values)

    return common_steps, np.array(aligned_data)


def plot_sample_efficiency(
    env_name: str,
    agents: list[str],
    eval_file: str = "episode_rewards",
    x_col: str = "steps",
    y_col: str = "rewards",
    step_interval: int = 1000,
    min_step: int = 0,
    max_step: int = 60000,
    save_path: str = None,
    title: str = None,
):
    """
    Create sample efficiency plot with stratified bootstrap intervals using rliable.

    Parameters
    ----------
    env_name : str
        Environment name (e.g., "MiniGrid-DoorKey-6x6-v0").
    agents : list[str]
        List of agent names (e.g., ["dqn"] or ["dqn", "rnd_naive", "rnd_on_sample"]).
    eval_file : str
        Name of CSV file containing evaluation data.
    x_col : str
        Name of x-axis column.
    y_col : str
        Name of y-axis column.
    step_interval : int
        Step interval for data alignment.
    min_step : int
        Minimum step for interpolation.
    max_step : int
        Maximum step for interpolation.
    save_path : str, optional
        Where to save the plot. If None, shows plot.
    title : str, optional
        Plot title. If None, auto-generates title.
    """
    all_train_scores = {}
    common_steps = None

    # Load data for each agent
    for agent in agents:
        result_dirs = get_result_dirs(env_name, agent)
        if not result_dirs:
            print(f"No result directories found for {agent}")
            continue

        # Get files with neccessary data
        file_paths = [os.path.join(d, f"{eval_file}.csv") for d in result_dirs]

        # Load and align data
        steps, aligned_data = load_and_align_data(
            file_paths, x_col, y_col, step_interval, min_step, max_step
        )

        if len(aligned_data) == 0:
            print(f"No valid data found for {agent}")
            continue

        # Store data for this agent with custom name
        agent_display_name = {
            "dqn": "DQN",
            "rnd_naive": "Naive RND-DQN",
            "rnd_on_sample": "On-sample RND-DQN",
        }.get(agent, agent.upper())

        all_train_scores[agent_display_name] = aligned_data
        if common_steps is None:
            common_steps = steps

    if not all_train_scores:
        print("No valid data found for any agent")
        return

    # Define aggregation function (IQM as recommended)
    iqm = lambda scores: np.array(
        [
            metrics.aggregate_iqm(scores[:, eval_idx])
            for eval_idx in range(scores.shape[-1])
        ]
    )

    # Get stratified bootstrap confidence intervals
    iqm_scores, iqm_cis = get_interval_estimates(
        all_train_scores,
        iqm,
        reps=2000,
    )

    # Create the plot
    plt.figure(figsize=(10, 6))
    plot_sample_efficiency_curve(
        common_steps,
        iqm_scores,
        iqm_cis,
        algorithms=list(all_train_scores.keys()),
        xlabel=x_col.title(),
        ylabel="IQM standard deviation",
        labelsize="x-large",
        ticklabelsize="x-large",
        marker="",
        linewidth=2,
        figsize=(8, 4),
    )

    if title:
        plt.title(title, fontsize="xx-large")
    else:
        # Auto-generate title
        agent_str = " vs ".join([agent.upper() for agent in agents])
        plt.title(f"{agent_str} - {env_name}", fontsize="xx-large")

    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=1000, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def get_result_dirs(env_name: str, agent: str) -> list[str]:
    """
    Get all result directories for a specific environment and agent.

    Parameters
    ----------
    env_name : str
        Name of the environment (e.g., "MiniGrid-DoorKey-6x6-v0").
    agent : str
        Agent type ("dqn", "rnd_naive", or "rnd_on_sample").

    Returns
    -------
    list[str]
        List of directory paths containing results for the specified agent.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    final_runs_dir = os.path.join(root, "../../results/final_runs")

    if not os.path.exists(final_runs_dir):
        print(f"Warning: Directory not found: {final_runs_dir}")
        return []

    pattern = f"{env_name}_{agent}_"
    result_dirs = []

    for dir_name in os.listdir(final_runs_dir):
        if dir_name.startswith(pattern):
            result_dirs.append(os.path.join(final_runs_dir, dir_name))

    print(f"Found {len(result_dirs)} result directories for {agent}")
    return result_dirs


if __name__ == "__main__":
    """
    Generate all comparison plots for DQN, RND-naive, and RND-on-sample agents.
    """
    env_name = "MiniGrid-DoorKey-6x6-v0"

    # Set path for resulting plots
    root = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(root, "../../results/plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Configuration for all plots
    plot_configs = [
        {
            "agents": ["rnd_naive", "rnd_on_sample", "dqn"],
            "eval_file": "episode_rewards",
            "y_col": "rewards",
            "filename": "rewards_over_steps",
            "title": "Episode rewards over steps",
        },
        {
            "agents": ["rnd_naive", "rnd_on_sample", "dqn"],
            "eval_file": "minibatch_values",
            "y_col": "extrinsic",
            "filename": "extrinsic",
            "title": "Mean extrinsic reward in sampled batches",
        },
        {
            "agents": ["rnd_naive", "rnd_on_sample"],  # Only RND agents
            "eval_file": "minibatch_values",
            "y_col": "intrinsic",
            "filename": "intrinsic",
            "title": "Mean intrinsic reward in sampled batches",
        },
        {
            "agents": ["rnd_naive", "rnd_on_sample", "dqn"],
            "eval_file": "minibatch_values",
            "y_col": "loss",
            "filename": "td_error",
            "title": "TD Error",
        },
        {
            "agents": ["rnd_naive", "rnd_on_sample", "dqn"],
            "eval_file": "minibatch_values",
            "y_col": "td_std",
            "filename": "td_std",
            "title": "Standard deviation of TD error",
        },
    ]

    # Generate all plots
    for config in plot_configs:
        plot_sample_efficiency(
            env_name=env_name,
            agents=config["agents"],
            eval_file=config["eval_file"],
            y_col=config["y_col"],
            save_path=os.path.join(plots_dir, f"{env_name}_{config['filename']}.png"),
            title=config["title"],
        )
