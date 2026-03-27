import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("pulse1.csv")

print("Missing Values Before Handling:")
print(df.isnull().sum())

# 🔥 Convert all columns to numeric safely
for col in ['Duration', 'Pulse', 'Maxpulse', 'Calories']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values
df['Calories'].fillna(df['Calories'].mean(), inplace=True)

print("\nData Types After Fix:")
print(df.dtypes)


# Outlier removal function
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print(f"\n{column} → Lower: {lower:.2f}, Upper: {upper:.2f}")

    return df[(df[column] >= lower) & (df[column] <= upper)]


# 📊 Before cleaning
plt.figure()
sns.boxplot(y=df['Duration'])
plt.title("Duration — Before Cleaning")
plt.show()


# Apply outlier removal
df_clean = df.copy()

for col in ['Duration', 'Pulse', 'Maxpulse', 'Calories']:
    df_clean = remove_outliers(df_clean, col)


print("\nRows before cleaning:", len(df))
print("Rows after cleaning:", len(df_clean))


# 📊 After cleaning
plt.figure()
sns.boxplot(y=df_clean['Duration'])
plt.title("Duration — After Cleaning")
plt.show()
