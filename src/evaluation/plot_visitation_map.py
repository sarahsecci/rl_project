import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

file_path = os.path.dirname(os.path.abspath(__file__))
run_folder = "MiniGrid-Empty-6x6-v0_rnd_naive_run_2025-07-27T14:40:46.729099/visitation_map.csv"  # xx
run_path = os.path.join(file_path, "../../results", run_folder)
df = pd.read_csv(run_path)

plt.figure(figsize=(6, 6))
sns.heatmap(df, cmap="viridis")
plt.title("Visitation Heatmap")
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()  # Align with MiniGrid
plt.show()
