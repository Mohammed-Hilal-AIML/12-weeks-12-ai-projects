# 🏠 House Price Prediction with Linear Regression

**Week 02 · AI Project Curriculum · Beginner**

A machine learning project that predicts California house prices using structured numerical data and regression algorithms.

## 📌 Project Overview

This project uses the **California Housing Dataset** provided by scikit-learn.

The goal is to predict median house value using housing characteristics such as median income, house age, rooms, bedrooms, population, occupancy, latitude, and longitude.

Three regression algorithms are trained and compared:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression

The model with the highest **R² score** on the test dataset is selected automatically.

## 🎯 Learning Objectives

- Supervised learning
- Regression vs. classification
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Feature engineering
- Categorical feature encoding
- Feature scaling
- Train/test splitting
- Linear, Ridge, and Lasso Regression
- MAE, RMSE, and R² evaluation
- Residual analysis
- Model saving/loading with Joblib
- Interactive prediction

## 🗂️ Project Structure

```text
Week_02_House_Price_Prediction/
├── house_price_prediction.py
├── requirements.txt
├── README.md
├── .gitignore
├── correlation_heatmap.png
├── actual_vs_predicted.png
├── residuals_distribution.png
├── house_price_model.pkl
├── scaler.pkl
└── feature_columns.pkl
```

The PNG and PKL files are generated when the script runs. They are included in this ZIP only when generated; `.gitignore` is configured so generated files can be excluded from GitHub if desired.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Joblib | Model persistence |

## 📊 Dataset

The project uses the **California Housing Dataset** built into scikit-learn.

The target values are represented in units of **$100,000**.

Examples:

```text
2.5 = $250,000
3.5 = $350,000
5.0 = $500,000
```

## ⚙️ Feature Engineering

Two additional features are created.

### Rooms per Person

```python
rooms_per_person = AveRooms / AveOccup
```

### Age Category

House age is divided into:

- Young
- Middle
- Old

The categorical feature is then converted into dummy variables for the regression models.

## 🔄 Data Preprocessing

The dataset is split into:

```text
80% → Training
20% → Testing
```

`StandardScaler` is fitted on the training data and then applied to both training and testing data.

## 🤖 Machine Learning Models

### Linear Regression

```python
LinearRegression()
```

### Ridge Regression

```python
Ridge(alpha=1.0)
```

### Lasso Regression

```python
Lasso(alpha=0.1)
```

Models are evaluated using:

- MAE
- RMSE
- R² Score

The highest-R² model is selected automatically.

## 📈 Visualizations

The script generates:

### Correlation Heatmap

`correlation_heatmap.png`

Shows correlations between the dataset features and target.

### Actual vs. Predicted

`actual_vs_predicted.png`

Compares actual house values with predictions from the selected model.

### Residual Distribution

`residuals_distribution.png`

Shows the distribution of prediction errors.

## 💾 Model Persistence

The trained model and preprocessing objects are saved with Joblib:

```text
house_price_model.pkl
scaler.pkl
feature_columns.pkl
```

They can be loaded later without retraining the model.

## 🧮 Interactive Prediction

After training, the script starts an interactive predictor.

Enter median income in units of **$10,000**:

```text
Enter Median Income in $10,000s:
3.5
```

The remaining features use baseline values calculated from the dataset.

Type `exit` or `quit` to stop the predictor.

## 🚀 Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Week_02_House_Price_Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python house_price_prediction.py
```

## 📋 Workflow

```text
Load Dataset
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Train Linear/Ridge/Lasso
     ↓
Evaluate Models
     ↓
Select Best Model
     ↓
Residual Analysis
     ↓
Save Model & Scaler
     ↓
Interactive Prediction
```

## ⚠️ Important Note

This project is intended as a machine learning learning exercise. Predictions should not be treated as professional real-estate valuations.

## 🎓 Project Context

**Program:** AI Project Curriculum  
**Level:** Beginner  
**Week:** 02 of 12  
**Project:** House Price Prediction with Linear Regression  
**Estimated Duration:** 5–7 hours

## 👨‍💻 Author

**Mohammed Hilal**

This project was created as part of a beginner machine learning project curriculum.

---

⭐ If you found this project useful, consider giving the repository a star!
