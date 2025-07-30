import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

file_path = os.path.dirname(os.path.abspath(__file__))
run_folder = "MiniGrid-DoorKey-6x6-v0_rnd_seed_0/visitation_map_50000.csv"  # xx
run_path = os.path.join(file_path, "../../results/runs", run_folder)
save_path = os.path.join(
    file_path, "../../results/runs/MiniGrid-DoorKey-6x6-v0_rnd_seed_0"
)
df = pd.read_csv(run_path)

plt.figure(figsize=(6, 6))
sns.heatmap(df, cmap="viridis")
plt.title("Visitation Heatmap")
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()  # Align with MiniGrid
# plt.show()
plt.savefig(save_path)
