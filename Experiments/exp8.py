import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset
df = pd.read_csv("pulse_output.csv")

print("Original Data:\n", df.head())

# 2. Handle missing values
df['Calories'] = df['Calories'].fillna(df['Calories'].mean())
df['Date'] = df['Date'].ffill()

# 3. Convert Date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 4. Remove any remaining NaN rows (important for PCA)
df = df.dropna()

# 5. Select numerical features
features = ['Duration','Pulse','Maxpulse','Calories']
X = df[features]

# 6. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# 8. Create dataframe
pca_df = pd.DataFrame(principal_components, columns=['PC1','PC2'])

# 9. Explained variance
print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Captured:", sum(pca.explained_variance_ratio_))

# 10. Visualization
plt.figure(figsize=(6,4))
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization (Pulse Dataset)")
plt.show()