import pandas as pd
from scipy.stats import shapiro, kruskal, f_oneway
import pingouin as pg

# Load the CSV file
algorithms={"decision_tree", "naive_bayes", "logistic_regression", "random_forest", "svm"}
for algorithm in algorithms:
    file_path = f"experiment_data/{algorithm}.csv"  # Update with correct path
    df = pd.read_csv(file_path, delimiter=";")

    print(f"\nAlgorithm: {algorithm}")
    # Convert numeric columns (handle commas in decimal values)
    for col in ["net_energy", "elapsed_time", "accuracy"]:
        df[col] = df[col].astype(float)

    # Check normality
    print("Normality Test Results:")
    normality_results = {}
    for col in ["net_energy", "elapsed_time", "accuracy"]:
        stat, p = shapiro(df[col])
        normality_results[col] = p
        print(f"{col}: p-value = {p}")

    # ANOVA if normal, Kruskal-Wallis if not
    if all(p >= 0.05 for p in normality_results.values()):
        print("\nUsing ANOVA:")
        for col in ["net_energy", "elapsed_time", "accuracy"]:
            stat, p = f_oneway(
                df[df["language"] == "cpp"][col],
                df[df["language"] == "python"][col],
                df[df["language"] == "julia"][col]
            )
            print(f"{col} ANOVA p-value: {p}")

            # Effect size
            effect_size = pg.anova(data=df, dv=col, between="language")["np2"].values[0]
            print(f"{col} Effect Size (η²): {effect_size}")

    else:
        print("\nUsing Kruskal-Wallis Test:")
        for col in ["net_energy", "elapsed_time", "accuracy"]:
            stat, p = kruskal(
                df[df["language"] == "cpp"][col],
                df[df["language"] == "python"][col],
                df[df["language"] == "julia"][col]
            )
            print(f"{col} Kruskal-Wallis p-value: {p}")
