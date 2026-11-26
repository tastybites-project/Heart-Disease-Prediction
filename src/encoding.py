import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

print("========== CATEGORICAL ENCODING ==========")

# Categorical columns
categorical_columns = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

print("\nCategorical Columns:")
print(categorical_columns)

# Display unique values before encoding
print("\nUnique Values Before Encoding:")

for column in categorical_columns:
    print(f"{column}: {df[column].unique()}")

# One-hot encoding
df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print("\nDataset Shape After Encoding:")
print(df_encoded.shape)

print("\nColumns After Encoding:")
print(df_encoded.columns.tolist())

# Save encoded dataset
df_encoded.to_csv(
    "dataset/heart_disease_encoded.csv",
    index=False
)

print("\nEncoded dataset saved successfully!")