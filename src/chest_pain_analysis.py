
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: LOAD DATASET
df = pd.read_csv("dataset/heart_disease_encoded.csv")

# STEP 2: CREATE DISEASE STATUS
df["disease_status"] = (df["num"] > 0).astype(int)

# STEP 3: RESTORE CHEST PAIN TYPE FROM ONE-HOT COLUMNS
chest_pain_columns = ["cp_1", "cp_2", "cp_3", "cp_4"]

df["chest_pain_type"] = (
    df[chest_pain_columns]
    .idxmax(axis=1)
    .str.replace("cp_", "", regex=False)
    .astype(int)
)

# STEP 4: CHEST PAIN TYPE LABELS
chest_pain_labels = {
    1: "Typical Angina",
    2: "Atypical Angina",
    3: "Non-Anginal Pain",
    4: "Asymptomatic"
}

df["chest_pain_label"] = df["chest_pain_type"].map(
    chest_pain_labels
)

# STEP 5: PATIENT COUNT
print("========== CHEST PAIN TYPE ANALYSIS ==========")

print("\nPatient Count by Chest Pain Type:")
print(df["chest_pain_label"].value_counts())

# STEP 6: HEART DISEASE DISTRIBUTION
disease_counts = pd.crosstab(
    df["chest_pain_label"],
    df["disease_status"]
)

disease_counts = disease_counts.reindex(
    list(chest_pain_labels.values()),
    fill_value=0
)

disease_counts.columns = [
    "No Heart Disease",
    "Heart Disease"
]

print("\nHeart Disease Distribution by Chest Pain Type:")
print(disease_counts)

# STEP 7: CREATE BAR CHART
disease_counts.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of Patients")
plt.xticks(rotation=20)
plt.legend(title="Disease Status")
plt.tight_layout()

# STEP 8: SAVE CHART
plt.savefig(
    "reports/chest_pain_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

# STEP 9: DISPLAY CHART
plt.show()

print("\nChest Pain Analysis completed!")
print("Chart saved: reports/chest_pain_analysis.png")
