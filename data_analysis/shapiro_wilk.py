from scipy.stats import shapiro
import numpy as np
import pandas as pd 

df = pd.read_csv("experiment_data/decision_tree.csv", delimiter=";")  

data = df[df["language"] == "cpp"]
data = data.replace(',', '.', regex=True).astype(float)

numerical_columns = ["total_energy", "elapsed_time", "accuracy"]

for col in numerical_columns:
    stat, p_value = shapiro(data[col])
    print(f"Shapiro-Wilk Test for {col}: Statistic={stat}, p-value={p_value}")

    # Interpretation
    if p_value > 0.05:
        print(f"✅ {col} is normally distributed (p-value = {p_value})\n")
    else:
        print(f"❌ {col} is NOT normally distributed (p-value = {p_value})\n")