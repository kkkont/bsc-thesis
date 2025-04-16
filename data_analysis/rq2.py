import pandas as pd

# Output file
output_file = "data_analysis/results/rq2.txt"

# Load and clean the merged dataset
df = pd.read_csv("experiment_data/merged_data.csv", delimiter=";")

# Convert comma decimals to dot and cast to float
for col in ["energy-cores", "energy-gpu", "energy-pkg", "energy-psys", 
            "total_energy", "elapsed_time", "accuracy", "net_energy"]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

# Clean algorithm names for readability
df["algorithm"] = df["algorithm"].str.replace("_", " ").str.title()

# List of metrics to analyze
metrics = ["net_energy", "elapsed_time", "accuracy"]

# Open output file for writing
with open(output_file, "w") as f:
    # Loop through each language and perform analysis
    for language in df["language"].unique():
        print(f"\n\n### Analyzing Language: {language} ###", file=f)
        
        # Filter data for the current language
        subset_language = df[df["language"] == language]
        
        # Compare metrics across different algorithms for this language
        print(f"\nAverage values for {language}:", file=f)
        grouped_means = subset_language.groupby("algorithm")[metrics].mean()
        print(grouped_means.round(4), file=f)
        
        # Spearman correlation for each algorithm
        for algorithm in subset_language["algorithm"].unique():
            print(f"\nSpearman Correlation — Algorithm: {algorithm}", file=f)
            subset_algorithm = subset_language[subset_language["algorithm"] == algorithm]
            corr = subset_algorithm[metrics].corr(method="spearman")
            print(corr.round(3), file=f)

print(f"\n✅ Spearman correlation results saved to: {output_file}")
