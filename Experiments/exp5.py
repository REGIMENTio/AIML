import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("pulse.csv")

print("Dataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# plt.figure(figsize=(8,5))
# sns.histplot(df['Duration'], bins=15, kde=True)
# plt.title("Duration Distribution")
# plt.show()

# plt.figure(figsize=(8,5))
# sns.histplot(df['Pulse'], bins=15, kde=True)
# plt.title("Pulse Distribution")
# plt.show()

# plt.figure(figsize=(8,5))
# sns.histplot(df['Maxpulse'], bins=15, kde=True)
# plt.title("Max Pulse Distribution")
# plt.show()

# plt.figure(figsize=(8,5))
# sns.histplot(df['Calories'], bins=15, kde=True)
# plt.title("Calories Distribution")
# plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Matrix")
plt.show()