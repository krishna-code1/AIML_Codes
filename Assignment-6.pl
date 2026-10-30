

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")


# 2. UPLOAD DATASET

print("Upload Mall_Customers.csv")
uploaded = files.upload()

file_name = next(iter(uploaded))
df = pd.read_csv(file_name)

print("\nDataset loaded successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# 3. DATA CLEANING

df = df.drop_duplicates()
df = df.dropna()

print("\nShape after cleaning:", df.shape)


# 4. EXPLORATORY DATA ANALYSIS

print("\nStatistical Summary:")
print(df.describe())

plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], kde=True)
plt.title("Customer Age Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["Annual Income (k$)"], kde=True)
plt.title("Annual Income Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["Spending Score (1-100)"], kde=True)
plt.title("Spending Score Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Gender"
)
plt.title("Annual Income vs Spending Score")
plt.show()


# 5. FEATURE SELECTION

# Use numerical customer behaviour features

X = df[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

print("\nSelected Features:")
print(X.head())


# 6. FEATURE SCALING

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling completed.")


# 7. ELBOW METHOD FOR K-MEANS

inertia_values = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_values.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(
    range(2, 11),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(range(2, 11))
plt.show()


# 8. SELECT BEST K USING SILHOUETTE SCORE

k_scores = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    k_scores.append(score)

    print(f"K = {k}, Silhouette Score = {score:.4f}")

best_k = range(2, 11)[np.argmax(k_scores)]

print("\nBest K based on Silhouette Score:", best_k)


# 9. K-MEANS CLUSTERING

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(
    X_scaled,
    df["KMeans_Cluster"]
)

kmeans_db = davies_bouldin_score(
    X_scaled,
    df["KMeans_Cluster"]
)

print("\nK-MEANS RESULTS")
print("Silhouette Score:", round(kmeans_silhouette, 4))
print("Davies-Bouldin Index:", round(kmeans_db, 4))


# K-Means Visualization

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=df["Annual Income (k$)"],
    y=df["Spending Score (1-100)"],
    hue=df["KMeans_Cluster"],
    palette="viridis",
    s=70
)

plt.title("K-Means Customer Segmentation")
plt.show()


# 10. HIERARCHICAL CLUSTERING

hierarchical = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

df["Hierarchical_Cluster"] = hierarchical.fit_predict(
    X_scaled
)

hier_silhouette = silhouette_score(
    X_scaled,
    df["Hierarchical_Cluster"]
)

hier_db = davies_bouldin_score(
    X_scaled,
    df["Hierarchical_Cluster"]
)

print("\nHIERARCHICAL CLUSTERING RESULTS")
print("Silhouette Score:", round(hier_silhouette, 4))
print("Davies-Bouldin Index:", round(hier_db, 4))


# Hierarchical Visualization

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=df["Annual Income (k$)"],
    y=df["Spending Score (1-100)"],
    hue=df["Hierarchical_Cluster"],
    palette="Set2",
    s=70
)

plt.title("Hierarchical Customer Segmentation")
plt.show()


# 11. DBSCAN CLUSTERING

# Find a suitable epsilon using nearest-neighbour distances

neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X_scaled)

distances, indices = neighbors.kneighbors(X_scaled)

k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 5))
plt.plot(k_distances)

plt.title("DBSCAN K-Distance Graph")
plt.xlabel("Data Points Sorted by Distance")
plt.ylabel("5th Nearest Neighbour Distance")
plt.show()


# Try multiple epsilon values and select the best valid result

best_dbscan = None
best_dbscan_score = -1
best_eps = None
best_min_samples = None

for eps in np.arange(0.3, 1.51, 0.1):

    for min_samples in [4, 5, 6]:

        model = DBSCAN(
            eps=round(eps, 2),
            min_samples=min_samples
        )

        labels = model.fit_predict(X_scaled)

        # Exclude noise points (-1) for evaluation

        mask = labels != -1
        valid_labels = labels[mask]
        valid_data = X_scaled[mask]

        unique_clusters = np.unique(valid_labels)

        if (
            len(unique_clusters) >= 2
            and len(valid_data) > len(unique_clusters)
        ):

            score = silhouette_score(
                valid_data,
                valid_labels
            )

            if score > best_dbscan_score:

                best_dbscan_score = score
                best_dbscan = model
                best_eps = round(eps, 2)
                best_min_samples = min_samples


# Fit selected DBSCAN model

if best_dbscan is not None:

    dbscan = DBSCAN(
        eps=best_eps,
        min_samples=best_min_samples
    )

    df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

    mask = df["DBSCAN_Cluster"] != -1

    db_labels = df.loc[mask, "DBSCAN_Cluster"]
    db_data = X_scaled[mask]

    db_unique = np.unique(db_labels)

    dbscan_silhouette = silhouette_score(
        db_data,
        db_labels
    )

    dbscan_db = davies_bouldin_score(
        db_data,
        db_labels
    )

    print("\nDBSCAN RESULTS")
    print("Best Epsilon:", best_eps)
    print("Best Min Samples:", best_min_samples)
    print("Number of Clusters:", len(db_unique))
    print("Noise Points:", sum(df["DBSCAN_Cluster"] == -1))
    print("Silhouette Score:", round(dbscan_silhouette, 4))
    print("Davies-Bouldin Index:", round(dbscan_db, 4))

else:

    df["DBSCAN_Cluster"] = -1

    dbscan_silhouette = np.nan
    dbscan_db = np.nan

    print("\nDBSCAN did not find at least two valid clusters.")
    print("Try different eps or min_samples values.")


# DBSCAN Visualization

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=df["Annual Income (k$)"],
    y=df["Spending Score (1-100)"],
    hue=df["DBSCAN_Cluster"],
    palette="tab10",
    s=70
)

plt.title("DBSCAN Customer Segmentation (-1 = Noise)")
plt.show()


# 12. COMPARISON OF CLUSTERING ALGORITHMS

results = pd.DataFrame({
    "Algorithm": [
        "K-Means",
        "Hierarchical",
        "DBSCAN"
    ],

    "Silhouette Score": [
        kmeans_silhouette,
        hier_silhouette,
        dbscan_silhouette
    ],

    "Davies-Bouldin Index": [
        kmeans_db,
        hier_db,
        dbscan_db
    ]
})

print("\nCLUSTERING PERFORMANCE COMPARISON")
print(results.round(4).to_string(index=False))


# Comparison Graph: Silhouette Score

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results,
    x="Algorithm",
    y="Silhouette Score"
)

plt.title("Silhouette Score Comparison")
plt.show()


# Comparison Graph: Davies-Bouldin Index

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results,
    x="Algorithm",
    y="Davies-Bouldin Index"
)

plt.title("Davies-Bouldin Index Comparison")
plt.show()


# 13. CUSTOMER SEGMENT ANALYSIS

print("\nK-MEANS CUSTOMER SEGMENT DETAILS")

segment_summary = df.groupby("KMeans_Cluster")[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean().round(2)

print(segment_summary)


# Number of customers in each cluster

print("\nCustomers in Each K-Means Cluster:")

print(
    df["KMeans_Cluster"]
    .value_counts()
    .sort_index()
)


# 14. FINAL RESULT ANALYSIS

print("\n========== FINAL RESULT ANALYSIS ==========")

print(
    "K-Means Silhouette Score:",
    round(kmeans_silhouette, 4)
)

print(
    "Hierarchical Silhouette Score:",
    round(hier_silhouette, 4)
)

if not np.isnan(dbscan_silhouette):

    print(
        "DBSCAN Silhouette Score:",
        round(dbscan_silhouette, 4)
    )

else:

    print("DBSCAN Silhouette Score: Not available")


valid_results = results.dropna(
    subset=["Silhouette Score"]
)

if len(valid_results) > 0:

    best_algorithm = valid_results.loc[
        valid_results["Silhouette Score"].idxmax(),
        "Algorithm"
    ]

    print(
        "\nBest algorithm based on Silhouette Score:",
        best_algorithm
    )

print("\nAssignment implementation completed successfully!")


# 15. SAVE FINAL DATASET WITH CLUSTER LABELS

df.to_csv(
    "Customer_Segmentation_Results.csv",
    index=False
)

files.download("Customer_Segmentation_Results.csv")

print("\nClustered dataset downloaded successfully.")
