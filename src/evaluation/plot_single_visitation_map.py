"""
Plot generation for mapping single state visitation maps.
Authors: Clara Schindler and Sarah Secci
Date: 09-08-25
Parts of this code were made with the help of Copilot
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

root = "../.."
file_path = os.path.dirname(os.path.abspath(__file__))
# run_folder = "../../results/sweeps/MiniGrid-DoorKey-8x8-v0_dqn_seed_1/18"
run_file = (
    "results/sweeps/MiniGrid-DoorKey-8x8-v0_dqn_seed_1/18/visitation_map_120000.csv"
)
run_path = os.path.join(file_path, root, run_file)
save_path = os.path.join(
    file_path, "../../results/sweeps/MiniGrid-DoorKey-8x8-v0_dqn_seed_1/18/visit.png"
)
df = pd.read_csv(run_path)

# plot state visitation map
plt.figure(figsize=(6, 6))
sns.heatmap(df, cmap="viridis")
plt.title("Visitation Heatmap")

plt.savefig(save_path)
