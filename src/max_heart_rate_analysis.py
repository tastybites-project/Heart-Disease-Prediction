
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

# STEP 3: MAXIMUM HEART RATE STATISTICS
print("========== MAXIMUM HEART RATE ANALYSIS ==========")

print("\nOverall Maximum Heart Rate Statistics:")
print(df["thalach"].describe().round(2))

# STEP 4: AVERAGE HEART RATE BY DISEASE STATUS
average_heart_rate = df.groupby("disease_label")["thalach"].mean()

print("\nAverage Maximum Heart Rate:")
print(average_heart_rate.round(2))

# STEP 5: CREATE BAR CHART
plt.figure(figsize=(8, 6))

average_heart_rate.plot(kind="bar")

plt.title("Average Maximum Heart Rate vs Heart Disease")
plt.xlabel("Heart Disease Status")
plt.ylabel("Average Maximum Heart Rate")
plt.xticks(rotation=0)
plt.tight_layout()

# STEP 6: SAVE CHART
plt.savefig(
    "reports/max_heart_rate_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

# STEP 7: DISPLAY CHART
plt.show()

print("\nMaximum Heart Rate Analysis completed!")
print("Chart saved: reports/max_heart_rate_analysis.png")
