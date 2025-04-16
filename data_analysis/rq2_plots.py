import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Label Mappings ===
metric_labels = {
    "net_energy": "Energy Consumption (Joules)",
    "elapsed_time": "Elapsed Time (ms)",
    "accuracy": "Accuracy (%)"
}

language_labels = {
    "cpp": "C++",
    "python": "Python",
    "julia": "Julia"
}

# === Load and clean data ===
df = pd.read_csv("experiment_data/merged_data.csv", delimiter=";")
for col in ["energy-cores", "energy-gpu", "energy-pkg", "energy-psys", 
            "total_energy", "elapsed_time", "accuracy", "net_energy"]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

df["algorithm"] = df["algorithm"].str.replace("_", " ").str.title()
df["elapsed_time"] = df["elapsed_time"] * 1000
df["accuracy"] = df["accuracy"] * 100 

# Metric pairs for scatter plots
metric_pairs = [("net_energy", "elapsed_time"),
                ("net_energy", "accuracy"),
                ("elapsed_time", "accuracy")]

# === Main Loop ===
for language in df["language"].unique():
    pretty_language = language_labels.get(language, language.title())
    subset_language = df[df["language"] == language]

    # --- Boxplots per metric ---
    for metric in metric_labels.keys():
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=subset_language, x="algorithm", y=metric, palette="pastel")
        plt.title(f"{metric_labels[metric]} Distribution — {pretty_language}", fontsize=16)
        plt.ylabel(metric_labels[metric])
        plt.xlabel("Algorithm")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"plots/rq2/distribution/{metric}_distribution_{language}.png", dpi=300)
        plt.close()

    # --- Heatmaps and Scatter Plots ---
    for algorithm in subset_language["algorithm"].unique():
        subset_algo = subset_language[subset_language["algorithm"] == algorithm]

        # --- Spearman Heatmap ---
        corr = subset_algo[list(metric_labels.keys())].corr(method="spearman")
        plt.figure(figsize=(5, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True,
                    linewidths=0.5, cbar_kws={"label": "Spearman Correlation"})
        plt.title(f"Spearman — {algorithm} ({pretty_language})")
        plt.tight_layout()
        plt.savefig(f"plots/rq2/spearman_correlation/spearman_{algorithm.lower().replace(' ', '_')}_{language}.png", dpi=300)
        plt.close()

        # --- Scatter Plots for Metric Pairs ---
        for metric_x, metric_y in metric_pairs:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=subset_algo,
                x=metric_x, y=metric_y,
                color="teal", s=100
            )
            plt.title(f"{algorithm} ({pretty_language}):\n{metric_labels[metric_x]} vs {metric_labels[metric_y]}", fontsize=15)
            plt.xlabel(metric_labels[metric_x])
            plt.ylabel(metric_labels[metric_y])
            plt.tight_layout()
            plt.savefig(f"plots/rq2/scatter_plots_metrics/{algorithm.lower().replace(' ', '_')}_{language}_{metric_x}_vs_{metric_y}.png", dpi=300)
            plt.close()
