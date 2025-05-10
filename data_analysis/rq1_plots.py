import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

df = pd.read_csv("experiment_data/merged_data.csv", delimiter=";")

for col in ["energy-cores", "energy-gpu", "energy-pkg", "energy-psys", "total_energy", "elapsed_time", "accuracy", "net_energy"]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

df["algorithm"] = df["algorithm"].str.replace("_", " ").str.title()
df["elapsed_time"] = df["elapsed_time"] * 1000
df["accuracy"] = df["accuracy"] * 100 

sns.set(style="whitegrid", font_scale=1.1)
metrics = ["net_energy", "elapsed_time", "accuracy"]
metric_labels = {
    "net_energy": "Energy Consumption (Joules)",
    "elapsed_time": "Elapsed Time (ms)",
    "accuracy": "Accuracy (%)"
}

for metric in metrics:
    algorithms = df["algorithm"].unique()
    n = len(algorithms)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*4), constrained_layout=True)
    axes = axes.flatten()
    palette = sns.color_palette("pastel")

    for i, algo in enumerate(algorithms):
        ax = axes[i]
        subset = df[df["algorithm"] == algo]
        sns.boxplot(data=subset, x="language", y=metric, palette=palette, linewidth=1.5, ax=ax)
        ax.set_title(algo)
        ax.set_xlabel("")
        ax.set_ylabel(metric_labels[metric])
        ax.set_xticklabels(["mlpack\n(C++)", "MLJ.jl\n(Julia)", "scikit-learn\n(Python)"])

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{metric_labels[metric]} by Library (Language) Across Algorithms", fontsize=18, fontweight="bold")

    ####### LEGEND #######

    language_order = ["mlpack", "MLJ.jl", "scikit-learn"]
    labels = ["mlpack (C++)", "MLJ.jl (Julia)", "scikit-learn (Python)"]
    handles = [Patch(color=palette[i], label=labels[i]) for i in range(len(labels))]
    fig.legend(handles=handles, loc='lower right',  title="Library (Language)")

    ######################

    filename = f"plots/rq1/{metric}_by_language.png"
    fig.savefig(filename, dpi=300)
    plt.close()
