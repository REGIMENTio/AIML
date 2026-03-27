import pandas as pd

df = pd.read_csv("pulse.csv")
df.head()

print("Shape of Dataset (Rows, Columns):", df.shape)
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nFirst Five Rows:")
print(df.head())

print("\nSize of Dataset:", df.size)

print("\n------->Missing Values in Each Column:<----------")
print(df.isnull().sum())

numeric_df = df.select_dtypes(include=['int64', 'float64'])

print("Number of Rows:", numeric_df.shape[0])
print("Number of Columns:", numeric_df.shape[1])

print("\nSum of Numerical Columns:")
print(numeric_df.sum())

print("\nAverage of Numerical Columns:")
print(numeric_df.mean())

print("\nMinimum Values:")
print(numeric_df.min())

print("\nMaximum Values:")
print(numeric_df.max())

df.to_csv("pulse_output.csv", index=False)

print("\nDataset exported successfully as 'pulse_output.csv'")