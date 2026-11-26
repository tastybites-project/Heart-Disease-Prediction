
import pandas as pd
import joblib
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# STEP 1: LOAD TRAINING DATA
X_train = pd.read_csv("dataset/X_train_scaled.csv")
y_train = pd.read_csv("dataset/y_train.csv").squeeze("columns")

# STEP 2: CREATE K-MEANS MODEL
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# STEP 3: TRAIN MODEL AND ASSIGN CLUSTERS
clusters = model.fit_predict(X_train)

# STEP 4: CALCULATE CLUSTER DISEASE RATIO
cluster_analysis = pd.DataFrame({
    "Cluster": clusters,
    "Disease Status": y_train.values
})

cluster_summary = cluster_analysis.groupby("Cluster").agg(
    Patient_Count=("Disease Status", "count"),
    Disease_Count=("Disease Status", "sum"),
    Disease_Ratio=("Disease Status", "mean")
).reset_index()

# STEP 5: ASSIGN RISK GROUPS
cluster_summary = cluster_summary.sort_values(
    "Disease_Ratio"
).reset_index(drop=True)

risk_labels = ["Low Risk", "Medium Risk", "High Risk"]

cluster_summary["Risk Category"] = risk_labels

risk_mapping = dict(
    zip(cluster_summary["Cluster"], cluster_summary["Risk Category"])
)

# STEP 6: CREATE PATIENT RESULTS
results = pd.DataFrame({
    "Patient_Number": range(1, len(X_train) + 1),
    "Cluster": clusters
})

results["Risk Category"] = results["Cluster"].map(risk_mapping)

print("========== K-MEANS CLUSTERING RESULTS ==========")

print("\nCluster Summary:")
print(cluster_summary.round(4))

print("\nPatient Risk Group Distribution:")
print(results["Risk Category"].value_counts())

# STEP 7: CALCULATE SILHOUETTE SCORE
score = silhouette_score(X_train, clusters)

print("\nSilhouette Score:", round(score, 4))

# STEP 8: SAVE MODEL AND RESULTS
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(model, "models/kmeans_model.pkl")

cluster_summary.to_csv(
    "reports/kmeans_cluster_summary.csv",
    index=False
)

results.to_csv(
    "reports/kmeans_patient_risk_groups.csv",
    index=False
)

# STEP 9: SAVE EVALUATION RESULT
metrics = pd.DataFrame({
    "Metric": ["Silhouette Score"],
    "Value": [score]
})

metrics.to_csv(
    "reports/kmeans_metrics.csv",
    index=False
)

print("\nK-Means Clustering Completed!")

print("Model saved: models/kmeans_model.pkl")
print("Cluster summary saved: reports/kmeans_cluster_summary.csv")
print("Patient groups saved: reports/kmeans_patient_risk_groups.csv")
print("Metrics saved: reports/kmeans_metrics.csv")

