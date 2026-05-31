import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("unemployment.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert Date column
# Remove spaces from date values
df["Date"] = df["Date"].astype(str).str.strip()

# Convert to datetime
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing values
df = df.ffill()

print(df.columns)

plt.figure(figsize=(12,6))
sns.lineplot(
    data=df,
    x="Date",
    y="Estimated Unemployment Rate (%)"
)
plt.title("Unemployment Rate Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------
# State-wise Unemployment
# ----------------------------------
plt.figure(figsize=(12,6))
state_avg = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
state_avg.plot(kind='bar')
plt.title('Average Unemployment Rate by Region')
plt.ylabel('Unemployment Rate (%)')
plt.tight_layout()
plt.show()

# ----------------------------------
# Heatmap
# ----------------------------------
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()
# Create Year column
df["Year"] = df["Date"].dt.year

# Average unemployment by year
covid_impact = df.groupby("Year")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(8,5))
covid_impact.plot(kind="bar")

plt.title("COVID-19 Impact on Unemployment")
plt.xlabel("Year")
plt.ylabel("Average Unemployment Rate (%)")

plt.tight_layout()
plt.show()