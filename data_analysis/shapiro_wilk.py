from scipy.stats import shapiro, zscore
import numpy as np
import pandas as pd 

# Initial shapiro-wilk test for normality on measurements to eliminate outliers

language = "cpp"
algorithm_file = "logistic_regression.csv"

df = pd.read_csv(f"experiment_data/{algorithm_file}", delimiter=";")  

data = df[df["language"] == language].copy()
print("Rows: ",len(data))

numerical_columns = ["total_energy", "elapsed_time", "accuracy"]

for col in numerical_columns:
    z_scores = np.abs(zscore(data[col]))  
    outlier_indices = np.where(z_scores > 3)[0]  # Find indices where Z-score > 3

    if len(outlier_indices) > 0:
        print(f"\n🔎 Outliers detected in '{col}':")
        for idx in outlier_indices:
            print(f"❌ Experiment {int(data.iloc[idx]['experiment_no'])} is an outlier in {col} (Z-score = {z_scores[idx]:.2f})")
    else:
        print(f"\n✅ No outliers detected in '{col}'.")

for col in numerical_columns:
    stat, p_value = shapiro(data[col])
    print(f"Shapiro-Wilk Test for {col}: Statistic={stat}, p-value={p_value}")

    if p_value > 0.05:
        print(f"✅ {col} is normally distributed (p-value = {p_value})\n")
    else:
        print(f"❌ {col} is NOT normally distributed (p-value = {p_value})\n")

