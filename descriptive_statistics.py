import pandas as pd

# Load dataset
df = pd.read_csv("nvidia_stock.csv")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Keep only relevant numeric columns
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

# Compute descriptive statistics
desc = pd.DataFrame({
    'Mean': df[numeric_cols].mean(),
    'Median': df[numeric_cols].median(),
    'Std Dev': df[numeric_cols].std(),
    'Min': df[numeric_cols].min(),
    '25% (Q1)': df[numeric_cols].quantile(0.25),
    '50% (Q2)': df[numeric_cols].quantile(0.50),
    '75% (Q3)': df[numeric_cols].quantile(0.75),
    'Max': df[numeric_cols].max()
})

# Add Interquartile Range (IQR)
desc['IQR'] = desc['75% (Q3)'] - desc['25% (Q1)']

# Round values for readability
desc = desc.round(4)

# Print table to console
print("\nDescriptive Statistics of NVIDIA Stock Data:\n")
print(desc)

# Save to CSV for including in report
desc.to_csv("descriptive_stats_table.csv")

print("\nDescriptive statistics saved to 'descriptive_stats_table.csv'")
