
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

# STEP 3: CALCULATE AVERAGE AGE
print("========== AGE VS HEART DISEASE ==========")

average_age = df.groupby("disease_label")["age"].mean()

print("\nAverage Age:")
print(average_age.round(2))

# STEP 4: CREATE BOXPLOT
plt.figure(figsize=(8, 6))

df.boxplot(
    column="age",
    by="disease_label"
)

plt.title("Age vs Heart Disease")
plt.suptitle("")
plt.xlabel("Heart Disease Status")
plt.ylabel("Age")
plt.tight_layout()

# STEP 5: SAVE CHART
plt.savefig(
    "reports/age_vs_heart_disease.png",
    dpi=300,
    bbox_inches="tight"
)

# STEP 6: DISPLAY CHART
plt.show()

print("\nAge vs Heart Disease Analysis completed!")
print("Chart saved: reports/age_vs_heart_disease.png")
