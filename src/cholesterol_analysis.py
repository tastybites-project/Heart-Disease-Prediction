
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: LOAD DATASET
df = pd.read_csv("dataset/heart_disease_encoded.csv")

# STEP 2: CREATE DISEASE STATUS
df["disease_status"] = (df["num"] > 0).astype(int)

df["disease_label"] = df["disease_status"].map({
    0: "No Heart Disease",
    1: "Heart Disease"
})

# STEP 3: CHOLESTEROL STATISTICS
print("========== CHOLESTEROL ANALYSIS ==========")

print("\nOverall Cholesterol Statistics:")
print(df["chol"].describe().round(2))

# STEP 4: AVERAGE CHOLESTEROL BY DISEASE STATUS
average_chol = df.groupby("disease_label")["chol"].mean()

print("\nAverage Cholesterol Level:")
print(average_chol.round(2))

# STEP 5: CREATE BAR CHART
plt.figure(figsize=(8, 6))

average_chol.plot(kind="bar")

plt.title("Average Cholesterol Level vs Heart Disease")
plt.xlabel("Heart Disease Status")
plt.ylabel("Average Cholesterol Level")
plt.xticks(rotation=0)
plt.tight_layout()

# STEP 6: SAVE CHART
plt.savefig(
    "reports/cholesterol_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

# STEP 7: DISPLAY CHART
plt.show()

print("\nCholesterol Analysis completed!")
print("Chart saved: reports/cholesterol_analysis.png")
