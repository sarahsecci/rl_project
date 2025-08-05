import glob
import math
import os

import gymnasium as gym

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from minigrid.wrappers import RGBImgObsWrapper


def vmap_quick(seed_start, seed_stop, agent, interval, cmap, show):
    """
    seed_start: seed from that plotting should start
    seed_stop: until which seed should be plottet
    agent: agent that was used for training -> Options: "dqn", "rnd_naive", "rnd_on_sample"
    seed: which training seed should be shown
    interval: interval of frames
    cmap: colourmap used for all plots
    show: if true shows plots instead of saving
    """
    root = "../.."
    file_path = os.path.dirname(os.path.abspath(__file__))
    current_interval = interval
    num_vmaps = int(60000 / interval)
    figures = []
    for i in range(seed_start, seed_stop):
        current_interval = interval
        for _ in range(num_vmaps):
            run_file = f"results/final_runs/MiniGrid-DoorKey-6x6-v0_{agent}_seed_{i}_time_*/visitation_map_{current_interval}.csv"
            run_path = os.path.join(file_path, root, run_file)
            matches = glob.glob(run_path)
            if not matches:
                print(f"Warning: No file found for pattern: {run_path}")
                current_interval += interval
                continue
            save_path = os.path.join(
                file_path,
                f"../../plots/visitation_maps/{agent}/MiniGrid-DoorKey-6x6-v0_{agent}_seed_{i}/visitation_map_{current_interval}.png",
            )
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)

            df = pd.read_csv(matches[0])

            if agent == "dqn":
                agent_name = "DQN"
            elif agent == "rnd_on_sample":
                agent_name = "on sample RND-DQN"
            elif agent == "rnd_naive":
                agent_name = "naive RND-DQN"
            else:
                agent_name = agent

            fig = plt.figure(num=f"Visitation {current_interval}", figsize=(6, 6))
            mask = df == 0
            sns.heatmap(
                df,
                cmap=cmap,
                mask=mask,
                square=True,
                vmin=0,
                vmax=16000,
                cbar_kws={"label": "Visitation Count", "shrink": 0.8},
            )
            plt.xticks([])  # Removes x-axis tick labels
            plt.yticks([])  # Removes y-axis tick labels
            plt.tick_params(left=False, bottom=False)
            plt.title(
                f"State visitation Map of {agent_name} after {current_interval} frames"
            )

            if show:
                figures.append(fig)
            else:
                plt.savefig(save_path)
                plt.close(fig)
            current_interval += interval

        # save initial setup of environment
        env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
        env = RGBImgObsWrapper(env)

        env.reset(seed=i)
        img = env.render()

        # Create save directory
        save_root = os.path.join(
            file_path,
            f"../../plots/visitation_maps/{agent}/MiniGrid-DoorKey-6x6-v0_{agent}_seed_{i}/",
        )
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, f"{agent}_seed_{i}_initial_setup.png")

        plt.imsave(save_path, img)
        env.close()

        if show:
            plt.show()


def plot_multiple_heatmaps(agent, seed, frames, max_plots, cols=3, cmap="rocket"):
    """
    agent: agent that was used for training -> Options: "dqn", "rnd_naive", "rnd_on_sample"
    seed: which training seed should be shown
    interval: interval of frames
    cols: number of columns in the grid
    cmap: colormap for all heatmaps
    """
    root = "../.."
    file_path = os.path.dirname(os.path.abspath(__file__))
    dfs = []
    titles = []
    # current_interval = interval
    # max_plots = int(60000/interval)

    for i in range(0, max_plots):
        run_file = f"results/final_runs/MiniGrid-DoorKey-6x6-v0_{agent}_seed_{seed}_time_*/visitation_map_{frames[i]}.csv"
        run_path = os.path.join(file_path, root, run_file)
        matches = glob.glob(run_path)
        if not matches:
            print(f"Warning: No file found for pattern: {run_path}")
            continue
        df = pd.read_csv(matches[0])
        dfs.append(df)
        if max_plots is not None:
            dfs = dfs[:max_plots]
        vmin = min(df.min().min() for df in dfs)
        vmax = max(df.max().max() for df in dfs)
        title = f"{frames[i]} Frames"
        titles.append(title)
        # current_interval += interval

    n = len(dfs)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.4]},
        squeeze=False,
        figsize=(15, 6),
    )  #

    for idx, (df, ax) in enumerate(zip(dfs, axes.flat)):
        mask = df == 0
        bar = False
        # if idx == 3: bar = True
        sns.heatmap(
            df,
            ax=ax,
            cmap=cmap,
            mask=mask,
            square=True,
            vmin=vmin,
            vmax=10000,
            cbar=bar,
            cbar_kws={"shrink": 0.2, "aspect": 10},
        )
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)

    # Add a single colorbar on the side
    # Use the last heatmap to create a mappable object

    # Create dummy heatmap to extract color scale
    cbar_ax = fig.add_axes([0.9, 0.29, 0.02, 0.4])  # [left, bottom, width, height]
    # sns.heatmap(dfs[0], cmap=cmap, cbar=True,
    #             square=True, vmin=vmin, vmax=15000)
    # plt.gcf().text(-0.02,0.45, "Y-axis", ha="center", va="center", rotation=90, fontsize=12)
    # Create a dummy mappable for the colorbar
    norm = plt.Normalize(vmin=vmin, vmax=10000)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar to the figure
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Visitation Count", fontsize=16)

    # Hide unused axes
    for ax in axes.flat[len(dfs) :]:
        ax.set_visible(False)

    if agent == "dqn":
        agent_name = "DQN"
    elif agent == "rnd_on_sample":
        agent_name = "on sample RND-DQN"
    elif agent == "rnd_naive":
        agent_name = "naive RND-DQN"
    else:
        agent_name = agent

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(
        f"State visitaion maps for {agent_name}, seed {seed}", fontsize=20, y=0.9
    )
    # plt.show()

    # Create save directory
    save_root = os.path.join(
        file_path,
        "../../plots/visitation_maps/",
    )
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, f"{agent}_seed_{seed}_.png")

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    # plot several state visitation maps as single files
    # vmap_quick(seed_start=0, seed_stop=21, agent="rnd_on_sample", interval=5000, cmap="inferno_r", show=False)

    frames = [5000, 10000, 20000, 40000]
    # plot several state visitation maps of one agent as a grid of subplots
    plot_multiple_heatmaps(
        agent="dqn", seed=12, frames=frames, max_plots=4, cols=5, cmap="inferno_r"
    )
