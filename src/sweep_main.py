import os
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import train_agent


def parse_performance_metric(results_dir: str) -> float:
    """
    Parse the final performance metric from training results.

    Parameters
    ----------
    results_dir : str
        Directory containing training results

    Returns
    -------
    float
        Performance metric for optimization (higher is better)
    """
    try:
        # Find episode rewards CSV
        episode_csv = os.path.join(results_dir, "episode_rewards.csv")

        if not os.path.exists(episode_csv):
            print(f"Episode rewards file not found: {episode_csv}")
            return -float("inf")

        # Read episode rewards
        df = pd.read_csv(episode_csv)

        if len(df) < 10:
            print(f"Too few episodes: {len(df)}")
            return -float("inf")

        # Compute performance metric
        rewards = df["rewards"].values

        # Option 1: Mean reward over last 20% of episodes
        last_20_percent = int(len(rewards) * 0.8)
        final_performance = np.mean(rewards[last_20_percent:])

        print(f"Final performance metric: {final_performance:.4f}")
        return float(final_performance)

    except Exception as e:
        print(f"Error parsing performance: {e}")
        return -float("inf")


@hydra.main(config_path="../config/", config_name="dqn_sweep", version_base="1.2")
def sweep_main(cfg: DictConfig) -> float:
    """
    Main function for hyperparameter sweeping.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration

    Returns
    -------
    float
        Performance metric for HyperSweeper optimization
    """
    try:
        # Get unique run identifier
        run_id = hydra.core.hydra_config.HydraConfig.get().job.num
        cfg.sweep.run_id = run_id

        print(f"Starting sweep trial {run_id}")
        print(f"Config: {cfg.agent}")

        # Train the agent
        results_dir = train_agent(cfg)

        # Parse performance metric
        performance = parse_performance_metric(results_dir)

        # Log trial result
        print(f"Trial {run_id} completed with performance: {performance:.4f}")

        return performance

    except Exception as e:
        print(f"Trial {run_id if 'run_id' in locals() else 'unknown'} failed: {e}")
        import traceback

        traceback.print_exc()
        return -float("inf")


if __name__ == "__main__":
    sweep_main()
