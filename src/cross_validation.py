import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


print("========== CROSS VALIDATION ==========")


# Load cleaned dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

print("\nDataset Shape:")
print(df.shape)


# Features and Target
X = df.drop("target", axis=1)
y = df["target"]


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# Logistic Regression Pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# 5-Fold Stratified Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Cross Validation
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)


print("\nCross Validation Scores:")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.4f}")


# Average score
mean_score = scores.mean()

print("\nMean Cross Validation Accuracy:")
print(f"{mean_score:.4f}")


# Standard deviation
std_score = scores.std()

print("\nStandard Deviation:")
print(f"{std_score:.4f}")


print("\nCross Validation completed successfully!")