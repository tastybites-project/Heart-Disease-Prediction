
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: LOAD DATASET
df = pd.read_csv("dataset/heart_disease_encoded.csv")

# STEP 2: CREATE AGE GROUPS
bins = [20, 30, 40, 50, 60, 70, 80]
labels = [
    "20-30",
    "31-40",
    "41-50",
    "51-60",
    "61-70",
    "71-80"
]

df["age_group"] = pd.cut(
    df["age"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# STEP 3: COUNT PATIENTS
age_counts = df["age_group"].value_counts().sort_index()

print("========== AGE GROUP DISTRIBUTION ==========")
print(age_counts)

# STEP 4: CREATE BAR CHART
plt.figure(figsize=(9, 6))

age_counts.plot(kind="bar")

plt.title("Patient Age Distribution")
plt.xlabel("Age Group")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()

# STEP 5: SAVE CHART
plt.savefig(
    "reports/age_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

# STEP 6: DISPLAY CHART
plt.show()

print("\nAge Distribution Chart created successfully!")
print("Chart saved: reports/age_distribution.png")
