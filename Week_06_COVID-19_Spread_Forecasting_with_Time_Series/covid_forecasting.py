from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

try:
    from pmdarima import auto_arima
except ImportError:
    auto_arima = None

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


DATA_FILE = Path("data/owid-covid-data.csv")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

START_DATE = "2020-01-01"
END_DATE = "2022-12-31"
WINDOW = 30
HORIZON = 7
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_india_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {DATA_FILE}\n"
            "Download the Our World in Data COVID-19 CSV and place it at:\n"
            "data/owid-covid-data.csv"
        )

    print("Loading COVID-19 data...")
    df = pd.read_csv(DATA_FILE)

    required = {"location", "date", "new_cases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    india = df[df["location"].eq("India")].copy()
    india["date"] = pd.to_datetime(india["date"], errors="coerce")
    india["new_cases"] = pd.to_numeric(india["new_cases"], errors="coerce")

    india = india.dropna(subset=["date", "new_cases"])
    india = india[(india["date"] >= START_DATE) & (india["date"] <= END_DATE)]
    india = india[["date", "new_cases"]].sort_values("date")
    india = india.set_index("date").asfreq("D")

    india["new_cases"] = india["new_cases"].fillna(0)
    india["rolling_7d"] = india["new_cases"].rolling(7).mean()
    india = india.dropna()

    return india


def stationarity_test(series, name):
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"\nADF Test - {name}")
    print(f"ADF statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.6f}")
    if result[1] < 0.05:
        print("Result: stationary at the 5% level.")
    else:
        print("Result: non-stationary at the 5% level.")
    return result[1]


def plot_data(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["new_cases"], label="Daily new cases", alpha=0.45)
    plt.plot(df.index, df["rolling_7d"], label="7-day rolling average")
    plt.title("India COVID-19 Daily New Cases (2020–2022)")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "india_daily_cases.png", dpi=150)
    plt.close()


def arima_forecast(train, horizon):
    if auto_arima is not None:
        print("\nSelecting ARIMA parameters with auto_arima...")
        model = auto_arima(
            train,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        order = model.order
        fitted = model
    else:
        print("\npmdarima not available; using ARIMA(1,1,1) fallback.")
        order = (1, 1, 1)
        fitted = ARIMA(train, order=order).fit()

    forecast = fitted.predict(n_periods=horizon) if auto_arima is not None else fitted.forecast(horizon)
    return np.asarray(forecast), order


def prophet_forecast(train_df, horizon):
    if Prophet is None:
        raise ImportError("Prophet is not installed.")

    prophet_df = train_df.reset_index()[["date", "new_cases"]].rename(
        columns={"date": "ds", "new_cases": "y"}
    )

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95,
    )

    # India public holidays are represented by common national holiday dates.
    holidays = pd.DataFrame({
        "holiday": [
            "republic_day", "independence_day", "gandhi_jayanti",
            "christmas"
        ],
        "ds": pd.to_datetime([
            "2020-01-26", "2020-08-15", "2020-10-02", "2020-12-25",
            "2021-01-26", "2021-08-15", "2021-10-02", "2021-12-25",
            "2022-01-26", "2022-08-15", "2022-10-02", "2022-12-25",
        ]),
        "lower_window": 0,
        "upper_window": 0,
    })
    model.holidays = holidays

    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    return model, forecast.tail(horizon)["yhat"].to_numpy()


def make_windows(values, window=30, horizon=7):
    X, y = [], []
    for i in range(len(values) - window - horizon + 1):
        X.append(values[i:i + window])
        y.append(values[i + window:i + window + horizon])
    return np.array(X), np.array(y)


def lstm_forecast(train_series, horizon=7):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()

    X, y = make_windows(scaled, WINDOW, horizon)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW, 1)),
        LSTM(32),
        Dense(horizon),
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X,
        y,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        shuffle=False,
    )

    last_window = scaled[-WINDOW:].reshape(1, WINDOW, 1)
    prediction_scaled = model.predict(last_window, verbose=0)[0]
    prediction = scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).flatten()
    prediction = np.maximum(prediction, 0)

    model.save(MODEL_DIR / "covid_lstm.keras")
    return prediction


def main():
    print("=" * 70)
    print("WEEK 06 - COVID-19 SPREAD FORECASTING WITH TIME SERIES")
    print("=" * 70)

    df = load_india_data()
    print(f"\nIndia records used: {len(df)}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

    plot_data(df)

    # Stationarity: raw 7-day rolling average, then first difference if needed.
    p_value = stationarity_test(df["rolling_7d"], "7-day rolling average")
    if p_value >= 0.05:
        differenced = df["rolling_7d"].diff().dropna()
        stationarity_test(differenced, "First difference of rolling average")

    # 80/20 chronological split.
    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    # For a directly comparable 30-day forecast, use the final 30 test observations.
    eval_horizon = min(30, len(test))
    actual_30 = test["new_cases"].iloc[:eval_horizon].to_numpy()

    print("\nTraining period:", train.index.min().date(), "to", train.index.max().date())
    print("Evaluation period:", test.index.min().date(), "to", test.index.max().date())

    # ARIMA
    arima_pred, arima_order = arima_forecast(
        train["new_cases"].to_numpy(), eval_horizon
    )
    print("ARIMA order:", arima_order)

    # Prophet
    prophet_model, prophet_pred = prophet_forecast(
        train.reset_index(), eval_horizon
    )

    # LSTM: forecast the same evaluation horizon in repeated 7-day blocks.
    lstm_predictions = []
    lstm_working = train["new_cases"].to_numpy().copy()
    remaining = eval_horizon
    while remaining > 0:
        block = lstm_forecast(lstm_working, HORIZON)
        take = min(remaining, HORIZON)
        lstm_predictions.extend(block[:take])
        lstm_working = np.concatenate([lstm_working, block[:take]])
        remaining -= take

    lstm_pred = np.array(lstm_predictions)

    # MAPE can be distorted by zero actuals; use a small epsilon.
    def safe_mape(actual, predicted):
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        denom = np.maximum(np.abs(actual), 1.0)
        return np.mean(np.abs((actual - predicted) / denom)) * 100

    metrics = pd.DataFrame({
        "model": ["ARIMA", "Prophet", "LSTM"],
        "MAPE_percent": [
            safe_mape(actual_30, arima_pred),
            safe_mape(actual_30, prophet_pred),
            safe_mape(actual_30, lstm_pred),
        ],
    })

    metrics.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(metrics.to_string(index=False))

    # Plot all forecasts against actual values.
    dates = test.index[:eval_horizon]
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual_30, label="Actual")
    plt.plot(dates, arima_pred, label="ARIMA")
    plt.plot(dates, prophet_pred, label="Prophet")
    plt.plot(dates, lstm_pred, label="LSTM")
    plt.title("COVID-19 Forecast Comparison — India")
    plt.xlabel("Date")
    plt.ylabel("Daily new cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast_comparison.png", dpi=150)
    plt.close()

    # Final 7-day LSTM forecast for the surge alert.
    final_7 = lstm_forecast(df["new_cases"].to_numpy(), HORIZON)
    current_7_avg = df["new_cases"].tail(7).mean()
    forecast_7_avg = final_7.mean()
    increase_pct = ((forecast_7_avg - current_7_avg) / max(current_7_avg, 1)) * 100

    alert = "SURGE ALERT" if increase_pct >= 50 else "No surge alert"

    alert_df = pd.DataFrame({
        "metric": [
            "current_7_day_average",
            "forecast_7_day_average",
            "forecast_increase_percent",
            "alert",
        ],
        "value": [
            current_7_avg,
            forecast_7_avg,
            increase_pct,
            alert,
        ],
    })
    alert_df.to_csv(OUTPUT_DIR / "surge_alert.csv", index=False)

    print("\n" + "=" * 70)
    print("7-DAY SURGE ALERT")
    print("=" * 70)
    print(f"Current 7-day average: {current_7_avg:,.2f}")
    print(f"Forecast 7-day average: {forecast_7_avg:,.2f}")
    print(f"Forecast increase: {increase_pct:.2f}%")
    print(f"Status: {alert}")

    print("\nOutputs saved in:", OUTPUT_DIR.resolve())
    print("Project completed.")


if __name__ == "__main__":
    main()
