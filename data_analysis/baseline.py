import numpy as np
import pandas as pd 

# Read the CSV file
data = pd.read_csv("experiment_data/baseline.csv", delimiter=";")  

data["total_energy"] = data["total_energy"].str.replace(',', '.').astype(float)

# Calculate average total energy consumption
avg_total_energy = data["total_energy"].mean()

# Assume measurement duration is 60 seconds
measurement_duration = 60

# Calculate idle power (Watts) for total energy
avg_total_power = avg_total_energy / measurement_duration

# Display results
print("Baseline Total Energy Consumption (Joules):")
print(avg_total_energy)
print("\nBaseline Idle Power (Watts):")
print(avg_total_power)
