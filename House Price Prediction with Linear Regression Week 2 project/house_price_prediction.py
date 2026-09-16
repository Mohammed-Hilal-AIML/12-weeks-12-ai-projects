import os
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Starting Week 2: House Price Prediction...", flush=True)

print("\n--- Step 1: Loading Dataset ---", flush=True)
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"Dataset shape: {df.shape}", flush=True)
print("\nFirst 5 rows:")
print(df.head())

print("\n--- Step 2: Exploratory Data Analysis ---", flush=True)
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()
print("Saved correlation heatmap to 'correlation_heatmap.png'", flush=True)

top_corr = correlation['MedHouseVal'].abs().sort_values(ascending=False)[1:4]
print("\nTop 3 features most correlated with price:")
print(top_corr)

print("\n--- Step 3: Feature Engineering ---", flush=True)
df['rooms_per_person'] = df['AveRooms'] / df['AveOccup']

df['age_category'] = pd.cut(
    df['HouseAge'],
    bins=[0, 15, 35, np.inf],
    labels=['Young', 'Middle', 'Old']
)

df = pd.get_dummies(df, columns=['age_category'], drop_first=True)
print("New features created successfully!", flush=True)

print("\n--- Step 4: Preprocessing & Scaling ---", flush=True)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Step 5: Training Models ---", flush=True)
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1)
}

best_model = None
best_r2 = -float('inf')
best_model_name = ""

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n{name}:")
    print(f"  MAE: ${mae * 100000:,.2f}")
    print(f"  RMSE: ${rmse * 100000:,.2f}")
    print(f"  R2 Score: {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

print(f"\nBest Model selected: {best_model_name}", flush=True)

best_preds = best_model.predict(X_test_scaled)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($100k)')
plt.ylabel('Predicted Price ($100k)')
plt.title(f'Actual vs Predicted ({best_model_name})')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

print("\n--- Step 6: Residual Analysis ---", flush=True)
residuals = y_test - best_preds

plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals (Error) Distribution')
plt.xlabel('Error Value ($100k)')
plt.tight_layout()
plt.savefig('residuals_distribution.png')
plt.close()

test_results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': best_preds,
    'Error': residuals
})

print("\nTop 5 Most Over-Predicted Houses (Predicted > Actual):")
print(test_results.sort_values(by='Error').head(5))

print("\nTop 5 Most Under-Predicted Houses (Actual > Predicted):")
print(test_results.sort_values(by='Error', ascending=False).head(5))

print("\n--- Step 7: Saving Model & Scaler ---", flush=True)
joblib.dump(best_model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
print("Model saved to 'house_price_model.pkl'", flush=True)

loaded_model = joblib.load('house_price_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('feature_columns.pkl')

print("\n" + "=" * 50)
print(" House Price Predictor Ready!")
print(" Enter median income to test (or type 'exit' to quit)")
print("=" * 50 + "\n")

baseline_sample = X.mean().to_dict()

while True:
    try:
        user_input = input(
            "Enter Median Income in $10,000s (e.g. 3.5 for $35,000): "
        )

        if user_input.lower() in ['exit', 'quit']:
            print("Exiting Week 2 project. Great work!")
            break

        income = float(user_input)

        sample_data = baseline_sample.copy()
        sample_data['MedInc'] = income

        sample_df = pd.DataFrame([sample_data])[feature_cols]
        scaled_sample = loaded_scaler.transform(sample_df)

        predicted_val = loaded_model.predict(scaled_sample)[0]
        actual_dollars = predicted_val * 100000

        print(f"--> Estimated House Price: ${actual_dollars:,.2f}\n")

    except ValueError:
        print("Invalid input. Please enter a numerical value.\n")
