
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("pulse.csv")

print("Missing Values Before Handling:")
print(df.isnull().sum())

df['Calories'].fillna(df['Calories'].mean(), inplace=True)

def remove_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print(f"\n{column} → Lower: {lower:.2f}, Upper: {upper:.2f}")

    outliers = df[(df[column] < lower) | (df[column] > upper)]
    print(f"Outliers in {column}: {len(outliers)}")

    return df[(df[column] >= lower) & (df[column] <= upper)]


sns.boxplot(y=df['Duration'])
plt.title("Duration — Before")
plt.show()

df_clean = remove_outliers('Duration')
df_clean = remove_outliers('Pulse')
df_clean = remove_outliers('Maxpulse')
df_clean = remove_outliers('Calories')

print("\nRows after cleaning:", len(df_clean))

sns.boxplot(y=df_clean['Duration'])
plt.title("Duration — After")
plt.show()