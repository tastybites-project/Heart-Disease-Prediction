import pandas as pd
from sklearn.model_selection import train_test_split

print("========== TRAIN TEST SPLIT ==========")

# Load cleaned dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

print("\nDataset Shape:")
print(df.shape)

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Feature Shape:")
print(X_train.shape)

print("\nTesting Feature Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

print("\nTrain/Test Split completed successfully!")