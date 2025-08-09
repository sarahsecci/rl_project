"""
Hyperparameter sweep script using HyperSMAC for agent optimization.
Authors: Clara Schindler and Sarah Secci
Date: 09-08-25
Parts of this code were made with the help of Copilot
"""

import os
import sys

import hydra
from omegaconf import DictConfig

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_agent import parse_performance_metric, train_agent


@hydra.main(
    config_path="../config/", config_name="rnd_on_sample_sweep_smac", version_base="1.2"
)  # Changed config name
def sweep_main(cfg: DictConfig) -> float:
    """
    Main function for hyperparameter sweeping with HyperSMAC.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing sweep parameters and agent configuration.

    Returns
    -------
    float
        Performance metric for HyperSMAC optimization (higher is better).
        Returns -inf if trial fails.
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
