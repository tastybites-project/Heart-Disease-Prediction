
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: LOAD DATASET
df = pd.read_csv("dataset/heart_disease_encoded.csv")

# STEP 2: CREATE DISEASE STATUS
df["disease_status"] = (df["num"] > 0).astype(int)

# STEP 3: COUNT PATIENTS
disease_counts = df["disease_status"].value_counts().sort_index()

labels = ["No Heart Disease", "Heart Disease"]
values = [
    disease_counts.get(0, 0),
    disease_counts.get(1, 0)
]

# STEP 4: CREATE PIE CHART
plt.figure(figsize=(8, 6))

plt.pie(
    values,
    labels=labels,
    autopct="%1.2f%%",
    startangle=90,
    explode=(0, 0.1)
)

plt.title("Heart Disease Distribution")
plt.axis("equal")

# STEP 5: SAVE CHART
plt.savefig(
    "reports/heart_disease_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

# STEP 6: DISPLAY CHART
plt.show()

print("Heart Disease Distribution Chart created successfully!")
print("Chart saved: reports/heart_disease_distribution.png")
