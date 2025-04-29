import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load & clean data ---
df = pd.read_csv("experiment_data/merged_data.csv", delimiter=";")

for col in ["energy-cores", "energy-gpu", "energy-pkg", "energy-psys", "total_energy", "elapsed_time", "accuracy", "net_energy"]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

df["algorithm"] = df["algorithm"].str.replace("_", " ").str.title()
df["elapsed_time"] = df["elapsed_time"] * 1000
df["accuracy"] = df["accuracy"] * 100 

# --- Plot settings ---
sns.set(style="whitegrid", font_scale=1.1)

metrics = ["net_energy", "elapsed_time", "accuracy"]

# Define labels with units
metric_labels = {
    "net_energy": "Energy Consumption (Joules)",
    "elapsed_time": "Elapsed Time (ms)",
    "accuracy": "Accuracy (%)"
}

for metric in metrics:
    g = sns.catplot(
        data=df,
        x="language", y=metric,
        col="algorithm",
        kind="box",
        col_wrap=3,
        height=4,
        sharey=False,
        palette="pastel",
        linewidth=1.5,
        fliersize=3,
        boxprops=dict(alpha=0.8),
    )

    for ax in g.axes.flatten():
        sns.swarmplot(
            data=df[df["algorithm"] == ax.get_title()],
            x="language", y=metric,
            color=".25", size=2, ax=ax
        )

    g.set_titles("{col_name}")
    g.set_axis_labels("", metric_labels[metric])
    g.set_xticklabels(["mlpack\n(C++)", "MLJ.jl\n(Julia)", "scikit-learn\n(Python)"])


    g.fig.suptitle(f"{metric_labels[metric]} by Language Across Algorithms", fontsize=18, fontweight="bold")
    g.fig.subplots_adjust(top=0.88)
 # Save to file
    filename = f"plots/rq1/{metric}_by_language.png"
    g.savefig(filename, dpi=300)
    plt.close()

