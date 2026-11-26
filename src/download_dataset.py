
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Download UCI Heart Disease Dataset
print("Downloading Heart Disease Dataset...")

heart_disease = fetch_ucirepo(id=45)

# Get features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Save dataset
df.to_csv("dataset/heart_disease.csv", index=False)

print("\nDataset downloaded successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset saved at:")
print("dataset/heart_disease.csv")
