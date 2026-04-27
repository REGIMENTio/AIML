import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("pulse_output.csv")

print("Original Data:\n", df.head())


df['Calories'] = df['Calories'].fillna(df['Calories'].mean())
df['Date'] = df['Date'].ffill()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 4. Remove duplicates
df = df.drop_duplicates()

df = df[df['Duration'] < 200]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Duration','Pulse','Maxpulse','Calories']])

scaled_df = pd.DataFrame(
    scaled_data,
    columns=['Duration','Pulse','Maxpulse','Calories']
)

plt.figure(figsize=(6,4))
plt.scatter(df['Duration'], df['Calories'])
plt.xlabel("Duration")
plt.ylabel("Calories")
plt.title("Duration vs Calories")
plt.show()

# 8. Output
print("\nCleaned Data:\n", df.head())
print("\nScaled Data:\n", scaled_df.head())

print("\nDataset Info:\n")
print(df.info())
