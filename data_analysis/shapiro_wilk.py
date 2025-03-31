from scipy.stats import shapiro, zscore
import numpy as np
import pandas as pd 

language = "python"
algorithm_file = "naive_bayes_v2.csv"

# Read the CSV file
df = pd.read_csv(f"experiment_data/{algorithm_file}", delimiter=";")  

# Take only rows where language is the one we are interested in
data = df[df["language"] == language].copy()
print("Rows: ",len(data))


# Numerical columns to check for outliers and normality
numerical_columns = ["total_energy", "elapsed_time", "accuracy"]
for col in numerical_columns:
    data[col] = data[col].str.replace(',', '.').astype(float)

for col in numerical_columns:
    z_scores = np.abs(zscore(data[col]))  # Compute Z-score for this column
    outlier_indices = np.where(z_scores > 3)[0]  # Find indices where Z-score > 3

    if len(outlier_indices) > 0:
        print(f"\n🔎 Outliers detected in '{col}':")
        for idx in outlier_indices:
            print(f"❌ Experiment {int(data.iloc[idx]['experiment_no'])} is an outlier in {col} (Z-score = {z_scores[idx]:.2f})")
    else:
        print(f"\n✅ No outliers detected in '{col}'.")

# Perform shapiro-wilk test for normality
for col in numerical_columns:
    stat, p_value = shapiro(data[col])
    print(f"Shapiro-Wilk Test for {col}: Statistic={stat}, p-value={p_value}")

    # Interpretation
    if p_value > 0.05:
        print(f"✅ {col} is normally distributed (p-value = {p_value})\n")
    else:
        print(f"❌ {col} is NOT normally distributed (p-value = {p_value})\n")

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Plot histogram
sns.histplot(data["elapsed_time"], kde=True, bins=10)
plt.title("Distribution of Accuracy")
plt.show()


