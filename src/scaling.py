import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load encoded dataset
df = pd.read_csv("dataset/heart_disease_encoded.csv")

print("========== FEATURE SCALING ==========")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# StandardScaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nScaled Features:")
print(X_scaled.head())

# Add target back
X_scaled["target"] = y.values

# Save scaled dataset
X_scaled.to_csv(
    "dataset/heart_disease_scaled.csv",
    index=False
)

print("\nScaled dataset saved successfully!")