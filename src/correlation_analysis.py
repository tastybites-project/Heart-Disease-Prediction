
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: LOAD DATASET
df = pd.read_csv("dataset/heart_disease_encoded.csv")

# STEP 2: CREATE BINARY TARGET
df["target"] = (df["num"] > 0).astype(int)

# STEP 3: SELECT ORIGINAL NUMERICAL FEATURES
selected_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "target"
]

correlation_data = df[selected_columns]

# STEP 4: CALCULATE CORRELATION
correlation_matrix = correlation_data.corr()

print("========== CORRELATION MATRIX ==========")
print(correlation_matrix.round(2))

# STEP 5: CREATE HEATMAP
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Heart Disease Feature Correlation Heatmap")
plt.tight_layout()

# STEP 6: SAVE HEATMAP
plt.savefig(
    "reports/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

# STEP 7: DISPLAY HEATMAP
plt.show()

print("\nCorrelation Analysis completed!")
print("Heatmap saved: reports/correlation_heatmap.png")
