
import pandas as pd

# STEP 1: LOAD ENCODED DATASET
df = pd.read_csv("dataset/heart_disease_encoded.csv")

# STEP 2: CREATE BINARY TARGET
df["target"] = (df["num"] > 0).astype(int)

# STEP 3: REMOVE ORIGINAL TARGET FROM FEATURES
X = df.drop(columns=["num", "target"])

# STEP 4: SELECT TARGET
y = df["target"]

# STEP 5: DISPLAY FEATURE INFORMATION
print("========== FEATURE SELECTION ==========")

print("\nFeature Dataset Shape:", X.shape)
print("Target Dataset Shape:", y.shape)

print("\nSelected Features:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts().sort_index())

# STEP 6: SAVE FEATURES AND TARGET
X.to_csv("dataset/X_features.csv", index=False)
y.to_csv("dataset/y_target.csv", index=False)

print("\nFeature Selection Completed!")
print("Features saved: dataset/X_features.csv")
print("Target saved: dataset/y_target.csv")
