import numpy as np
import pandas as pd 

# This script calculates the net energy consumption for each measurement

filename = "decision_tree.csv" # Replace according to the algorithm being analyzed :)

# Read the baseline measurements CSV file
data = pd.read_csv("experiment_data/baseline.csv", delimiter=";")  
data["total_energy"] = data["total_energy"].str.replace(',', '.').astype(float)

# Calculate baseline 
avg_total_energy = data["total_energy"].mean()
measurement_duration = 60
avg_total_power = avg_total_energy / measurement_duration

print("Baseline Total Energy Consumption (Joules):")
print(avg_total_energy)
print("\nBaseline Idle Power (Watts):")
print(avg_total_power)

# Read the experiment measurements CSV file
experiment_data = pd.read_csv(f"experiment_data/{filename}", delimiter=";")

# Calculate net energy for each measurement
experiment_data["net_energy"] = experiment_data["total_energy"] - (avg_total_power * experiment_data["elapsed_time"])

# Save the updated CSV file
experiment_data.to_csv(f"experiment_data/{filename}", sep=';', index=False)
