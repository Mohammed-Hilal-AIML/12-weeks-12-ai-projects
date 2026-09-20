# Week 06 — COVID-19 Spread Forecasting with Time Series

Time-series forecasting project based on the Week 06 AI Project Curriculum.

The project uses COVID-19 data for India and compares **ARIMA, Facebook Prophet, and an LSTM neural network**. It also performs an Augmented Dickey-Fuller stationarity test, calculates MAPE, produces forecast comparison charts, and includes a 7-day **SURGE ALERT** system.

## Project requirements

- Python 3.12
- pandas
- NumPy
- Matplotlib
- statsmodels
- pmdarima
- Prophet
- TensorFlow
- scikit-learn

## Project structure

```text
Week_06_COVID-19_Spread_Forecasting_with_Time_Series/
│
├── covid_forecasting.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   └── owid-covid-data.csv
│
├── models/
│   └── covid_lstm.keras
│
└── outputs/
    ├── india_daily_cases.png
    ├── forecast_comparison.png
    ├── model_comparison.csv
    └── surge_alert.csv
```

## 1. Download the dataset

Download the Our World in Data COVID-19 CSV.

Stable CSV:

https://covid.ourworldindata.org/data/owid-covid-data.csv

Place it here:

```text
data/owid-covid-data.csv
```

The dataset contains country/date records. The script automatically filters the data to **India** and uses the 2020–2022 period required by this project.

See `data/README.md` for detailed dataset instructions.

## 2. Install Python 3.12

Check:

```powershell
py -3.12 --version
```

You should see Python 3.12.x.

## 3. Install requirements

From the project folder:

```powershell
py -3.12 -m pip install -r requirements.txt
```

## 4. Run in VS Code

Open the project folder in VS Code.

Open:

**Terminal → New Terminal**

Then run:

```powershell
py -3.12 covid_forecasting.py
```

## What the program does

### Step 1 — COVID-19 data

Loads the Our World in Data COVID-19 dataset, filters for India, parses dates, and uses daily new cases from 2020 through 2022.

### Step 2 — Stationarity

Creates a 7-day rolling average and runs the Augmented Dickey-Fuller test. If the rolling average is non-stationary, first differencing is tested.

### Step 3 — ARIMA

Uses `auto_arima` to select non-seasonal `(p,d,q)` parameters and produces a forecast.

### Step 4 — Prophet

Uses Facebook/Meta Prophet with weekly and yearly seasonality and India-related holiday dates.

### Step 5 — LSTM

Creates 30-day sliding windows to predict the next 7 days using:

```text
LSTM(64)
   ↓
LSTM(32)
   ↓
Dense(7)
```

Early stopping is used during training.

### Step 6 — Model comparison

ARIMA, Prophet, and LSTM forecasts are compared with MAPE.

### Step 7 — Surge alert

The program compares the predicted 7-day average with the current 7-day average.

A:

```text
SURGE ALERT
```

is generated when the forecast shows a **50% or greater increase**.

## Output files

After a successful run:

```text
outputs/india_daily_cases.png
outputs/forecast_comparison.png
outputs/model_comparison.csv
outputs/surge_alert.csv
```

The trained LSTM is saved as:

```text
models/covid_lstm.keras
```

## GitHub

Do not upload the full COVID dataset or generated model to GitHub unless you specifically want to distribute them. The `.gitignore` excludes the dataset, model, and generated outputs.

Users can clone the repository, download the dataset, place it in `data/`, install the requirements, and run the Python script.

## Dataset

Our World in Data COVID-19 data:

https://ourworldindata.org/

Stable CSV:

https://covid.ourworldindata.org/data/owid-covid-data.csv

The Week 06 curriculum specifies Our World in Data COVID-19 data and an India-focused 2020–2022 analysis.
