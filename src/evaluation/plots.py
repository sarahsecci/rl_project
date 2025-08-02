import os
import re
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve


def extract_algorithm_from_folder(folder_name: str) -> str:
    """
    Extract algorithm name from folder name.
    Expected format: MiniGrid-{env}-{size}-v0_{algorithm}_seed_{seed}_time_{timestamp}
    """
    # Split by underscores and find the algorithm part
    parts = folder_name.split("_")

    # Look for known algorithms
    for i, part in enumerate(parts):
        if part == "dqn":
            return "dqn"
        elif part == "rnd":
            if i + 1 < len(parts):
                if parts[i + 1] == "naive":
                    return "rnd_naive"
                elif parts[i + 1] == "on":
                    return "rnd_on_sample"

    return "unknown"


def extract_env_from_folder(folder_name: str) -> str:
    """
    Extract environment name from folder name.
    Expected format: MiniGrid-{env}-{size}-v0_{algorithm}_seed_{seed}_time_{timestamp}
    """
    # Extract everything before the algorithm part
    match = re.match(r"(MiniGrid-[^_]+-\d+x\d+-v\d+)", folder_name)
    if match:
        return match.group(1)
    return "unknown_env"


def _plot_multiple_seeds(
    file_paths: list[str],
    x_col: str,
    y_cols: list[str],
    ylabel: str,
    algorithm: str,
    save_path: str,
    title: str = None,
):
    """Plot aggregated data for multiple seeds using rliable."""
    if not file_paths:
        print(f"No files found for algorithm {algorithm}")
        return

    n_seeds = len(file_paths)

    # Read data from different runs
    dfs = []
    min_length = float("inf")

    for i, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        df["seed"] = i
        dfs.append(df)
        min_length = min(min_length, len(df))

    if not dfs:
        print(f"No valid data files found for {algorithm}")
        return

    # Update n_seeds to actual number of valid files
    n_seeds = len(dfs)

    # Truncate all dataframes to the same length
    for i in range(len(dfs)):
        dfs[i] = dfs[i].head(min_length)

    # Combine the dataframes
    df = pd.concat(dfs, ignore_index=True)

    # Get x values
    x_values = df[x_col].to_numpy().reshape((n_seeds, -1))[0]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot each y column
    for i, y_col in enumerate(y_cols):
        if y_col not in df.columns:
            print(f"Warning: Column {y_col} not found in data")
            continue

        # Reshape data for rliable
        train_scores = {algorithm: df[y_col].to_numpy().reshape((n_seeds, -1))}

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
            algorithms=[algorithm],
            xlabel=x_col.title(),
            ylabel=f"IQM {y_col.replace('_', ' ').title()}",
            marker="",
            linewidth=1,
        )

    if title:
        plt.title(title)

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_rewards_over_steps(
    episode_files: list[str], algorithm: str, save_path: str, env_name: str
):
    """Plot episode rewards over steps."""
    title = f"{algorithm.upper()} - Episode Rewards over Steps ({env_name})"
    _plot_multiple_seeds(
        file_paths=episode_files,
        x_col="steps",
        y_cols=["rewards"],
        ylabel="Rewards",
        algorithm=algorithm,
        save_path=save_path,
        title=title,
    )


def plot_minibatch_rewards_over_steps(
    minibatch_files: list[str], algorithm: str, save_path: str, env_name: str
):
    """Plot minibatch rewards (extrinsic and intrinsic) over steps."""
    # Determine which columns to plot based on algorithm
    y_cols = ["extrinsic"]
    if algorithm.startswith("rnd"):
        y_cols.append("intrinsic")

    title = f"{algorithm.upper()} - Minibatch Rewards over Steps ({env_name})"
    _plot_multiple_seeds(
        file_paths=minibatch_files,
        x_col="steps",
        y_cols=y_cols,
        ylabel="Rewards",
        algorithm=algorithm,
        save_path=save_path,
        title=title,
    )


def plot_td_error_over_steps(
    minibatch_files: list[str], algorithm: str, save_path: str, env_name: str
):
    """Plot TD error (loss) and TD standard deviation over steps."""
    title = f"{algorithm.upper()} - TD Error and Std over Steps ({env_name})"
    _plot_multiple_seeds(
        file_paths=minibatch_files,
        x_col="steps",
        y_cols=["loss", "td_std"],
        ylabel="TD Metrics",
        algorithm=algorithm,
        save_path=save_path,
        title=title,
    )


def collect_files_by_algorithm_and_env(results_dir: str) -> dict:
    """
    Collect all files organized by algorithm and environment.
    Returns: {(algorithm, env): {'episode': [...], 'minibatch': [...]}}
    """
    files_by_algo_env = defaultdict(lambda: {"episode": [], "minibatch": []})

    for dir_name in os.listdir(results_dir):
        dir_path = os.path.join(results_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        algorithm = extract_algorithm_from_folder(dir_name)
        env_name = extract_env_from_folder(dir_name)

        if algorithm == "unknown":
            continue

        # Check for required files
        episode_file = os.path.join(dir_path, "episode_rewards.csv")
        minibatch_file = os.path.join(dir_path, "minibatch_values.csv")

        if os.path.exists(episode_file):
            files_by_algo_env[(algorithm, env_name)]["episode"].append(episode_file)
        if os.path.exists(minibatch_file):
            files_by_algo_env[(algorithm, env_name)]["minibatch"].append(minibatch_file)

    return files_by_algo_env


def generate_plots_for_algorithm_env(
    algorithm: str, env_name: str, results_dir: str = None, plots_base_dir: str = None
):
    """
    Generate all plots for a specific algorithm and environment combination.

    Args:
        algorithm: Algorithm name ('dqn', 'rnd_naive', 'rnd_on_sample')
        env_name: Environment name (e.g., 'MiniGrid-DoorKey-6x6-v0')
        results_dir: Directory containing result folders (optional)
        plots_base_dir: Base directory for saving plots (optional)
    """
    # Set default paths if not provided
    if results_dir is None:
        file_path = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(file_path, "../../results/runs")

    if plots_base_dir is None:
        plots_base_dir = os.path.join(results_dir, "plots")

    os.makedirs(plots_base_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    # Collect all files
    files_by_algo_env = collect_files_by_algorithm_and_env(results_dir)

    # Check if the specific combination exists
    key = (algorithm, env_name)
    if key not in files_by_algo_env:
        print(f"No data found for algorithm '{algorithm}' on environment '{env_name}'")
        print("Available combinations:")
        for (algo, env), files in files_by_algo_env.items():
            episode_count = len(files["episode"])
            minibatch_count = len(files["minibatch"])
            print(
                f"  - {algo} on {env}: {episode_count} episode files, {minibatch_count} minibatch files"
            )
        return

    files = files_by_algo_env[key]
    print(f"\nProcessing {algorithm} on {env_name}...")
    print(
        f"Found {len(files['episode'])} episode files and {len(files['minibatch'])} minibatch files"
    )

    # Create algorithm-specific plot directory
    algo_plot_dir = os.path.join(plots_base_dir, algorithm, env_name)

    # Plot 1: Episode rewards over steps
    if files["episode"]:
        save_path = os.path.join(algo_plot_dir, "episode_rewards_over_steps.png")
        plot_rewards_over_steps(files["episode"], algorithm, save_path, env_name)

    # Plot 2: Minibatch rewards over steps
    if files["minibatch"]:
        save_path = os.path.join(algo_plot_dir, "minibatch_rewards_over_steps.png")
        plot_minibatch_rewards_over_steps(
            files["minibatch"], algorithm, save_path, env_name
        )

    # Plot 3: TD error over steps
    if files["minibatch"]:
        save_path = os.path.join(algo_plot_dir, "td_error_over_steps.png")
        plot_td_error_over_steps(files["minibatch"], algorithm, save_path, env_name)

    print(f"Completed plots for {algorithm} on {env_name}")
    print(f"Plots saved in: {algo_plot_dir}")


def generate_all_plots(results_dir: str = None, plots_base_dir: str = None):
    """Generate plots for all available algorithm-environment combinations."""
    # Set default paths if not provided
    if results_dir is None:
        file_path = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(file_path, "../../results/runs")

    if plots_base_dir is None:
        plots_base_dir = os.path.join(results_dir, "plots")

    os.makedirs(plots_base_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    # Collect files by algorithm and environment
    files_by_algo_env = collect_files_by_algorithm_and_env(results_dir)

    if not files_by_algo_env:
        print("No valid algorithm directories found")
        return

    # Generate plots for each algorithm-environment combination
    for (algorithm, env_name), files in files_by_algo_env.items():
        generate_plots_for_algorithm_env(
            algorithm, env_name, results_dir, plots_base_dir
        )

    print(f"\nAll plots saved in: {plots_base_dir}")


def list_available_combinations(results_dir: str = None):
    """List all available algorithm-environment combinations."""
    if results_dir is None:
        file_path = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(file_path, "../../results/runs")

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    files_by_algo_env = collect_files_by_algorithm_and_env(results_dir)

    if not files_by_algo_env:
        print("No valid algorithm directories found")
        return

    print("Available algorithm-environment combinations:")
    for (algorithm, env_name), files in files_by_algo_env.items():
        episode_count = len(files["episode"])
        minibatch_count = len(files["minibatch"])
        print(
            f"  - {algorithm} on {env_name}: {episode_count} episode files, {minibatch_count} minibatch files"
        )


if __name__ == "__main__":
    # Example usage:

    # Option 1: Generate plots for a specific algorithm and environment
    generate_plots_for_algorithm_env("rnd_naive", "MiniGrid-DoorKey-6x6-v0")

    # Option 2: Generate plots for all available combinations
    # generate_all_plots()

    # Option 3: List available combinations first
    # list_available_combinations()

    # Option 4: Generate plots for multiple specific combinations
    # combinations = [
    #     ("dqn", "MiniGrid-DoorKey-6x6-v0"),
    #     ("rnd_naive", "MiniGrid-DoorKey-6x6-v0"),
    #     ("dqn", "MiniGrid-DoorKey-8x8-v0"),
    # ]
    #
    # for algorithm, env_name in combinations:
    #     generate_plots_for_algorithm_env(algorithm, env_name)
