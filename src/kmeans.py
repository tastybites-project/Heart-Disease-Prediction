import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load cleaned dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create 3 clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Add cluster column
df["cluster"] = clusters

# Show cluster distribution
print("\nCluster Distribution:")
print(df["cluster"].value_counts().sort_index())

# Compare clusters with heart disease target
print("\nCluster vs Heart Disease:")
print(pd.crosstab(df["cluster"], df["target"]))

# Calculate average heart disease rate for each cluster
cluster_risk = df.groupby("cluster")["target"].mean().sort_values()

print("\nCluster Risk Level:")
print(cluster_risk)

# Map clusters according to actual heart disease risk
risk_mapping = {}

for i, cluster in enumerate(cluster_risk.index):
    if i == 0:
        risk_mapping[cluster] = "Low Risk"
    elif i == 1:
        risk_mapping[cluster] = "Medium Risk"
    else:
        risk_mapping[cluster] = "High Risk"

df["risk_category"] = df["cluster"].map(risk_mapping)

# Display risk category distribution
print("\nRisk Category Distribution:")
print(df["risk_category"].value_counts())

# Save result
df.to_csv("dataset/heart_disease_clusters.csv", index=False)

print("\nK-Means clustering completed successfully.")
print("Saved to: dataset/heart_disease_clusters.csv")

# Visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    df["age"],
    df["thalach"],
    c=df["cluster"],
    cmap="viridis",
    s=50
)

plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.title("Patient Risk Clusters")

plt.colorbar(label="Cluster")
plt.show()