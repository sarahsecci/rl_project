import os
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_agent import train_agent


def parse_performance_metric(results_dir: str) -> float:
    """
    Parse the final performance metric from training results.
    Optimizes for minimal episode length (shorter episodes = better performance).

    Parameters
    ----------
    results_dir : str
        Directory containing training results

    Returns
    -------
    float
        Performance metric for optimization (higher is better)
    """
    # Constants
    MAX_EPISODE_STEPS = 640  # Maximum steps before truncation in MiniGrid

    try:
        # Find episode rewards CSV
        episode_csv = os.path.join(results_dir, "episode_rewards.csv")

        if not os.path.exists(episode_csv):
            print(f"Episode rewards file not found: {episode_csv}")
            return -float("inf")

        # Read episode data
        df = pd.read_csv(episode_csv)

        if len(df) < 10:
            print(f"Too few episodes: {len(df)}")
            return -float("inf")

        # Calculate episode lengths from step differences (current_step - previous_step)
        steps = df["steps"].values
        rewards = df["rewards"].values
        episode_lengths = []
        prev_step = 0

        for _, current_step in enumerate(steps):
            episode_length = current_step - prev_step
            episode_lengths.append(episode_length)
            prev_step = current_step

        episode_lengths = np.array(episode_lengths)

        # Get average episode length of last 25% of episodes
        if len(episode_lengths) < 4:
            print("Not enough episodes to calculate average length")
            return -float("inf")
        last_25_percent = int(len(episode_lengths) * 0.75)
        final_episode_lengths = episode_lengths[last_25_percent:]
        avg_episode_length = np.mean(final_episode_lengths)

        # Get amount of successful episodes
        successful_episodes = []

        for i, (length, reward) in enumerate(zip(episode_lengths, rewards)):
            # Consider episode successful if it completed in less than max steps
            if length < MAX_EPISODE_STEPS and reward > 0:
                successful_episodes.append(length)

        # Normalize by max steps
        performance = avg_episode_length / MAX_EPISODE_STEPS

        print(f"Successful episodes: {len(successful_episodes)}")
        print(f"Average episode length (successful): {avg_episode_length:.2f}")
        print(f"Performance metric (normalized length): {performance:.4f}")

        return float(performance)

    except Exception as e:
        print(f"Error parsing performance: {e}")
        return -float("inf")


@hydra.main(
    config_path="../config/", config_name="rnd_on_sample_sweep_smac", version_base="1.2"
)  # Changed config name
def sweep_main(cfg: DictConfig) -> float:
    """
    Main function for hyperparameter sweeping with HyperSMAC.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration

    Returns
    -------
    float
        Performance metric for HyperSMAC optimization (higher is better)
    """
    try:
        # Get unique run identifier
        run_id = hydra.core.hydra_config.HydraConfig.get().job.num
        cfg.sweep.run_id = run_id

        print(f"Starting SMAC trial {run_id}")
        print(f"Budget (num_frames): {cfg.train.num_frames}")
        print(f"Config: {cfg.agent}")

        # Train the agent
        results_dir = train_agent(cfg)

        # Parse performance metric
        performance = parse_performance_metric(results_dir)

        # Log trial result
        print(f"Trial {run_id} completed with performance: {performance:.4f}")
        print(f"Budget used: {cfg.train.num_frames} frames")

        return performance

    except Exception as e:
        print(f"Trial {run_id if 'run_id' in locals() else 'unknown'} failed: {e}")
        import traceback

        traceback.print_exc()
        return -float("inf")


if __name__ == "__main__":
    sweep_main()
