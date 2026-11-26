import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

print("========== DATA TYPES CHECK ==========")

# Show data types
print("\nData Types:")
print(df.dtypes)

# Show unique values
print("\nUnique Values:")
for column in df.columns:
    print(f"{column}: {df[column].unique()}")

# Show number of unique values
print("\nNumber of Unique Values:")
print(df.nunique())