# 🛍️ Customer Segmentation with K-Means Clustering

**Week 03 · AI Project Curriculum · Beginner**

A machine learning project that automatically groups mall customers into meaningful customer segments using **K-Means clustering**.

This project introduces unsupervised machine learning, where the algorithm discovers groups in the data without predefined target labels.

## 📌 Project Overview

The project uses the **Mall Customer Segmentation Dataset** and analyzes customers using:

- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

For clustering, the primary features are:

- **Annual Income (k$)**
- **Spending Score (1-100)**

K-Means is used to divide customers into **5 clusters**, following the typical K=5 choice described in the Week 3 curriculum.

## 🎯 Learning Objectives

This project covers:

- Unsupervised learning
- Supervised vs. unsupervised learning
- K-Means clustering
- Centroids
- Iterations and convergence
- StandardScaler preprocessing
- Elbow Method
- Inertia / within-cluster sum of squares
- Silhouette Score
- 2D cluster visualization
- Cluster centroid visualization
- Customer personas
- Business-oriented segmentation strategies

## 🗂️ Project Structure

```text
Week_03_Customer_Segmentation_KMeans/
│
├── customer_segmentation.py
├── Mall_Customers.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── income_vs_spending_by_gender.png
├── elbow_method.png
├── customer_clusters.png
│
├── cluster_summary.csv
├── segmented_customers.csv
└── business_report.txt
```

> `Mall_Customers.csv` is the dataset input. Download the free Mall Customer Segmentation dataset from Kaggle and place it in the project folder before running the program.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Scaling and K-Means |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |

## 📊 Dataset

The project uses the **Mall Customer Segmentation Data** from Kaggle.

The curriculum describes the dataset as containing **200 customers** with fields including CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1-100).

## 🔍 Step 1 — Load the Dataset

The program loads:

```text
Mall_Customers.csv
```

using Pandas and displays the dataset shape, first rows, and information.

## 👀 Step 2 — Visual Exploration

The project creates an income-versus-spending scatter plot with points differentiated by gender.

Generated file:

```text
income_vs_spending_by_gender.png
```

This provides an initial visual indication of possible customer groups before clustering.

## ⚙️ Step 3 — Preprocessing

The clustering features are:

```python
features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]
```

The features are standardized using:

```python
StandardScaler()
```

Gender is also encoded for later experiments:

```text
Male → 0
Female → 1
```

Gender is **not used in the main K-Means clustering**.

## 📐 Step 4 — Elbow Method

K-Means is evaluated for:

```text
K = 1 through 10
```

For each K, the model's inertia is recorded.

The results are plotted in:

```text
elbow_method.png
```

The Week 3 curriculum notes that **K=5 is typically the elbow choice for this dataset**, so the project uses five clusters.

## 🤖 Step 5 — K-Means Clustering

The final model uses:

```python
KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
```

A new `Cluster` column is added to the customer DataFrame.

The program also calculates and prints the cluster centroids.

Centroids are converted back from scaled values to the original income/spending units so they are easier to interpret.

## 📏 Silhouette Score

The project calculates the silhouette score to provide a numerical measure of cluster separation and cohesion.

A silhouette score closer to 1 generally indicates better-defined clusters, while lower values indicate more overlap between groups.

The exact score is calculated when the script runs on the dataset.

## 📊 Step 6 — Cluster Visualization

The final customer segments are displayed in a color-coded scatter plot.

Cluster centroids are marked with stars.

Generated file:

```text
customer_clusters.png
```

The program also creates descriptive segment names based on the relative income and spending characteristics of each cluster.

Example personas include:

- High Income High Spenders
- High Income Low Spenders
- Budget Friendly Spenders
- Low Income Low Spenders

## 💼 Step 7 — Business Report

The project automatically creates:

```text
business_report.txt
```

The report contains:

- Number of customers in each cluster
- Average income
- Average spending score
- A descriptive segment name
- A suggested marketing strategy

A CSV summary is also created:

```text
cluster_summary.csv
```

The complete customer dataset with cluster assignments is saved as:

```text
segmented_customers.csv
```

## 🚀 Installation

### 1. Download the project

Extract the ZIP file and open the project folder.

### 2. Download the dataset

Download the **Mall Customer Segmentation Data** from Kaggle.

Place:

```text
Mall_Customers.csv
```

inside the project directory.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python customer_segmentation.py
```

## 📋 Project Workflow

```text
Load Mall Customer Dataset
          ↓
Visual Exploration
          ↓
Select Income + Spending Score
          ↓
StandardScaler
          ↓
Elbow Method (K=1 to 10)
          ↓
Choose K=5
          ↓
Train K-Means
          ↓
Calculate Silhouette Score
          ↓
Add Cluster Labels
          ↓
Visualize Clusters + Centroids
          ↓
Create Customer Personas
          ↓
Generate Business Report
```

## 📦 Requirements

Install all required libraries with:

```bash
pip install -r requirements.txt
```

Required packages:

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
```

## 📁 Generated Files

| File | Description |
|---|---|
| `income_vs_spending_by_gender.png` | Initial customer visualization |
| `elbow_method.png` | Elbow Method plot |
| `customer_clusters.png` | Final K-Means visualization |
| `cluster_summary.csv` | Summary of each customer segment |
| `segmented_customers.csv` | Customers with cluster assignments |
| `business_report.txt` | Marketing/business interpretation |

## ⚠️ Important Note

This project is intended as a machine learning learning exercise.

The customer segment names and marketing strategies are interpretations based on the clustering results. They should be treated as analytical starting points rather than definitive descriptions of individual customers.

## 🎓 Project Context

**Program:** AI Project Curriculum  
**Level:** Beginner  
**Week:** 03 of 12  
**Project:** Customer Segmentation with K-Means Clustering  
**Estimated Duration:** 6–8 hours

## 👨‍💻 Author

**Mohammed Hilal**

This project was created as part of a beginner machine learning project curriculum.

---

⭐ If you found this project useful, consider giving the repository a star!
