import numpy as np
import pandas as pd 

filename = "decision_tree.csv"

# Read the baseline CSV file
data = pd.read_csv("experiment_data/baseline.csv", delimiter=";")  

data["total_energy"] = data["total_energy"].str.replace(',', '.').astype(float)
# Calculate average total energy consumption
avg_total_energy = data["total_energy"].mean()

# Measurement duration was 60 seconds
measurement_duration = 60

# Calculate idle power (Watts) for total energy
avg_total_power = avg_total_energy / measurement_duration

# Display baseline results
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
