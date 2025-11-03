# results_and_findings_graphs.py
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# === Load dataset ===
df = pd.read_csv("nvidia_stock.csv")

# Ensure 'Date' is parsed properly
df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce').dt.tz_localize(None)
df = df.sort_values('Date')

# === Create output folder ===
os.makedirs("figures", exist_ok=True)

# === Basic sanity check ===
cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df = df[cols + ['Date']].dropna()

# ================================================================
# 1️⃣ Price Trend Over Time
# ================================================================
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Open'], label='Open')
plt.plot(df['Date'], df['Close'], label='Close', alpha=0.7)
plt.title("NVIDIA Stock Price Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("figures/1_price_trend.png", dpi=300)
plt.close()

# ================================================================
# 2️⃣ Rolling Mean and Volatility (30-day)
# ================================================================
df['Rolling_Mean'] = df['Open'].rolling(window=30).mean()
df['Rolling_Std'] = df['Open'].rolling(window=30).std()

plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Rolling_Mean'], label='30-Day Rolling Mean')
plt.plot(df['Date'], df['Rolling_Std'], label='30-Day Rolling Std (Volatility)')
plt.title("Rolling Mean and Volatility of Open Prices")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("figures/2_rolling_mean_std.png", dpi=300)
plt.close()

# ================================================================
# 3️⃣ Daily Returns Distribution (Histogram)
# ================================================================
df['Daily_Return'] = df['Open'].pct_change() * 100

plt.figure(figsize=(8,5))
plt.hist(df['Daily_Return'].dropna(), bins=60)
plt.title("Distribution of Daily % Returns (Open Price)")
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/3_returns_hist.png", dpi=300)
plt.close()

# ================================================================
# 4️⃣ Autocorrelation of Returns (ACF)
# ================================================================
fig = sm.graphics.tsa.plot_acf(df['Daily_Return'].dropna(), lags=40)
plt.title("Autocorrelation of Daily Returns")
plt.tight_layout()
fig.savefig("figures/4_returns_acf.png", dpi=300)
plt.close()

# ================================================================
# 5️⃣ Correlation Heatmap between Features
# ================================================================
corr = df[cols].corr()

plt.figure(figsize=(6,5))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(cols)), cols, rotation=45)
plt.yticks(range(len(cols)), cols)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("figures/5_corr_heatmap.png", dpi=300)
plt.close()

# ================================================================
# 6️⃣ Volume vs Absolute Returns (Volatility relation)
# ================================================================
df['Abs_Return'] = df['Daily_Return'].abs()

plt.figure(figsize=(8,5))
plt.scatter(df['Volume'], df['Abs_Return'], alpha=0.5)
plt.xscale('log')
plt.title("Relationship Between Volume and Absolute Returns")
plt.xlabel("Volume (log scale)")
plt.ylabel("Absolute Daily Return (%)")
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("figures/6_volume_vs_return.png", dpi=300)
plt.close()

# ================================================================
# 7️⃣ Monthly Return Distribution (Boxplot)
# ================================================================
df['Month'] = df['Date'].dt.month
monthly_returns = [df.loc[df['Month']==m, 'Daily_Return'].dropna() for m in range(1,13)]

plt.figure(figsize=(10,5))
plt.boxplot(monthly_returns, labels=[str(m) for m in range(1,13)])
plt.title("Monthly Distribution of Daily % Returns")
plt.xlabel("Month")
plt.ylabel("Daily Return (%)")
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("figures/7_monthly_boxplot.png", dpi=300)
plt.close()

print("✅ All graphs saved in the 'figures/' folder successfully!")
