import pandas as pd
import numpy as np

csv_path = "nvidia_stock.csv"  # <- update
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['Date'])
df = df.sort_values('Date')

# Keep standard columns; rename if needed
cols = ['Open','High','Low','Close','Volume']
present = [c for c in cols if c in df.columns]
X = df[present].copy()

# Descriptive stats
q1 = X.quantile(0.25)
q3 = X.quantile(0.75)
iqr = q3 - q1

stats = pd.DataFrame({
    'count': X.count(),
    'mean': X.mean(),
    'median': X.median(),
    'std': X.std(ddof=1),
    'min': X.min(),
    'q1': q1,
    'q3': q3,
    'iqr': iqr,
    'max': X.max()
}).round(4)

stats.to_csv("descriptive_statistics.csv")
print(stats)
