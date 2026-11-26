import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

print("========== DATA VISUALIZATION ==========")

# 1. Target Distribution
plt.figure(figsize=(6, 4))

df["target"].value_counts().plot(
    kind="bar"
)

plt.title("Heart Disease Target Distribution")
plt.xlabel("Target")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# 2. Age Distribution
plt.figure(figsize=(7, 4))

df["age"].plot(
    kind="hist",
    bins=20
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()


# 3. Cholesterol Distribution
plt.figure(figsize=(7, 4))

df["chol"].plot(
    kind="hist",
    bins=20
)

plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()


# 4. Maximum Heart Rate Distribution
plt.figure(figsize=(7, 4))

df["thalach"].plot(
    kind="hist",
    bins=20
)

plt.title("Maximum Heart Rate Distribution")
plt.xlabel("Maximum Heart Rate")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()


print("\nVisualization completed successfully!")

# Correlation Heatmap
plt.figure(figsize=(12, 8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()