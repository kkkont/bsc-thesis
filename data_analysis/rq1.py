import pandas as pd
from scipy.stats import shapiro, kruskal, f_oneway
import pingouin as pg

# Output file path
output_file = "data_analysis/results/rq1.txt"

algorithms = {"decision_tree", "naive_bayes", "logistic_regression", "random_forest", "svm"}

with open(output_file, "w") as f:
    for algorithm in algorithms:
        print(f"\n\n====================== {algorithm.upper()} ======================", file=f)
        file_path = f"experiment_data/{algorithm}.csv"
        df = pd.read_csv(file_path, delimiter=";")

        # Convert necessary columns to float
        for col in ["net_energy", "elapsed_time", "accuracy"]:
            df[col] = df[col].astype(float)

        # Show group means for context
        print("\nAverage values per language:", file=f)
        print(df.groupby("language")[["net_energy", "elapsed_time", "accuracy"]].mean().round(4), file=f)

        # Perform analysis for each metric separately
        for col in ["net_energy", "elapsed_time", "accuracy"]:
            print(f"\n--- {col.replace('_', ' ').title()} ---", file=f)

            # Normality check
            stat, p_normal = shapiro(df[col])
            print(f"Shapiro-Wilk p-value: {p_normal}", file=f)

            if p_normal >= 0.05:
                print("✅ Data is normally distributed → Using ANOVA", file=f)
                stat, p_anova = f_oneway(
                    df[df["language"] == "cpp"][col],
                    df[df["language"] == "python"][col],
                    df[df["language"] == "julia"][col]
                )
                print(f"ANOVA p-value: {p_anova}", file=f)

                anova_result = pg.anova(data=df, dv=col, between="language")
                eta_sq = anova_result["np2"].values[0]
                print(f"Effect size (η²): {eta_sq:.4f}", file=f)
            else:
                print("❌ Data is not normally distributed → Using Kruskal-Wallis", file=f)
                stat, p_kruskal = kruskal(
                    df[df["language"] == "cpp"][col],
                    df[df["language"] == "python"][col],
                    df[df["language"] == "julia"][col]
                )
                print(f"Kruskal-Wallis p-value: {p_kruskal}", file=f)

                epsilon_sq = (stat - 3 + 1) / (len(df) - 1)
                print(f"Effect size (ε²): {epsilon_sq:.4f}", file=f)

print(f"\n✅ All results saved to: {output_file}")
