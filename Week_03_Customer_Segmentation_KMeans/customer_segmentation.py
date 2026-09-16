import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("Starting Week 3: Customer Segmentation with K-Means...", flush=True)

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================
print("\n--- Step 1: Loading Dataset ---", flush=True)

DATA_FILE = "Mall_Customers.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Download the Mall Customer Segmentation "
        "dataset from Kaggle and place the CSV in this project folder."
    )

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}", flush=True)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

# ==========================================
# STEP 2: VISUAL EXPLORATION
# ==========================================
print("\n--- Step 2: Visual Exploration ---", flush=True)

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Gender",
    s=80
)
plt.title("Annual Income vs Spending Score by Gender")
plt.tight_layout()
plt.savefig("income_vs_spending_by_gender.png", dpi=150)
plt.close()

# ==========================================
# STEP 3: PREPROCESSING
# ==========================================
print("\n--- Step 3: Preprocessing ---", flush=True)

features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode Gender as 0/1 for later experiments
df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})

print("Selected features:", features)
print("Features scaled successfully.", flush=True)

# ==========================================
# STEP 4: ELBOW METHOD
# ==========================================
print("\n--- Step 4: Finding Optimal K with Elbow Method ---", flush=True)

inertias = []
k_values = range(1, 11)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=150)
plt.close()

print("Elbow plot saved to 'elbow_method.png'.")

# Week 3 curriculum identifies k=5 as the typical choice for this dataset.
optimal_k = 5
print(f"Using K = {optimal_k}.", flush=True)

# ==========================================
# STEP 5: TRAIN K-MEANS
# ==========================================
print("\n--- Step 5: Training K-Means ---", flush=True)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

centroid_df = pd.DataFrame(
    centroids_original,
    columns=features
)
centroid_df.index.name = "Cluster"

print("\nCluster centroids:")
print(centroid_df)

silhouette = silhouette_score(X_scaled, df["Cluster"])
print(f"\nSilhouette Score: {silhouette:.4f}", flush=True)

# ==========================================
# STEP 6: VISUALIZE CLUSTERS
# ==========================================
print("\n--- Step 6: Visualizing Clusters ---", flush=True)

cluster_names = {
    0: "Segment 0",
    1: "Segment 1",
    2: "Segment 2",
    3: "Segment 3",
    4: "Segment 4"
}

# Automatically assign descriptive names based on centroid income/spending.
income_median = df["Annual Income (k$)"].median()
spending_median = df["Spending Score (1-100)"].median()

for cluster_id, row in centroid_df.iterrows():
    income = row["Annual Income (k$)"]
    spending = row["Spending Score (1-100)"]

    if income >= income_median and spending >= spending_median:
        name = "High Income High Spenders"
    elif income >= income_median and spending < spending_median:
        name = "High Income Low Spenders"
    elif income < income_median and spending >= spending_median:
        name = "Budget Friendly Spenders"
    else:
        name = "Low Income Low Spenders"

    cluster_names[cluster_id] = name

df["Segment"] = df["Cluster"].map(cluster_names)

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Segment",
    palette="tab10",
    s=90
)

plt.scatter(
    centroid_df["Annual Income (k$)"],
    centroid_df["Spending Score (1-100)"],
    marker="*",
    s=300,
    color="black",
    label="Centroids"
)

for cluster_id, row in centroid_df.iterrows():
    plt.annotate(
        f"C{cluster_id}",
        (row["Annual Income (k$)"], row["Spending Score (1-100)"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold"
    )

plt.title("Customer Segmentation using K-Means")
plt.tight_layout()
plt.savefig("customer_clusters.png", dpi=150)
plt.close()

# ==========================================
# STEP 7: BUSINESS REPORT
# ==========================================
print("\n--- Step 7: Business Report ---", flush=True)

summary = (
    df.groupby(["Cluster", "Segment"])
    .agg(
        Customers=("CustomerID", "count"),
        Avg_Income=("Annual Income (k$)", "mean"),
        Avg_Spending_Score=("Spending Score (1-100)", "mean")
    )
    .reset_index()
)

summary.to_csv("cluster_summary.csv", index=False)
df.to_csv("segmented_customers.csv", index=False)

with open("business_report.txt", "w", encoding="utf-8") as report:
    report.write("CUSTOMER SEGMENTATION BUSINESS REPORT\n")
    report.write("=" * 50 + "\n\n")
    report.write(
        "This report summarizes five customer segments created using "
        "K-Means clustering on Annual Income and Spending Score.\n\n"
    )

    for _, row in summary.iterrows():
        report.write(f"Cluster {int(row['Cluster'])}: {row['Segment']}\n")
        report.write(f"- Customers: {int(row['Customers'])}\n")
        report.write(f"- Average income: {row['Avg_Income']:.2f} k$\n")
        report.write(
            f"- Average spending score: {row['Avg_Spending_Score']:.2f}\n"
        )

        segment = row["Segment"]

        if segment == "High Income High Spenders":
            strategy = (
                "Use premium products, loyalty rewards, personalized offers, "
                "and early access campaigns."
            )
        elif segment == "High Income Low Spenders":
            strategy = (
                "Use targeted promotions, product education, introductory "
                "offers, and incentives designed to increase engagement."
            )
        elif segment == "Budget Friendly Spenders":
            strategy = (
                "Use discounts, bundles, value-oriented products, and "
                "frequent promotional campaigns."
            )
        else:
            strategy = (
                "Use affordable products, entry-level offers, seasonal "
                "discounts, and budget-focused promotions."
            )

        report.write(f"- Suggested strategy: {strategy}\n\n")

print("\nCluster summary:")
print(summary.to_string(index=False))

print("\nFiles generated:")
print("- income_vs_spending_by_gender.png")
print("- elbow_method.png")
print("- customer_clusters.png")
print("- cluster_summary.csv")
print("- segmented_customers.csv")
print("- business_report.txt")

print("\nWeek 3 project completed successfully!", flush=True)
