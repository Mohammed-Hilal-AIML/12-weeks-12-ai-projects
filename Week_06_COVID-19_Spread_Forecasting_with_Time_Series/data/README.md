# COVID-19 Dataset

This folder contains the input data for the Week 06 forecasting project.

## Required file

```text
owid-covid-data.csv
```

Place it directly in this folder:

```text
data/owid-covid-data.csv
```

## Download

Our World in Data provides a stable CSV URL:

https://covid.ourworldindata.org/data/owid-covid-data.csv

Download the CSV and save it as:

```text
owid-covid-data.csv
```

## Expected structure

```text
data/
├── README.md
└── owid-covid-data.csv
```

## What the script uses

The program:

1. Loads the CSV.
2. Filters `location == "India"`.
3. Parses the `date` column.
4. Uses `new_cases`.
5. Limits the analysis to 2020–2022.
6. Builds the time-series forecasting models.

Do not rename the CSV unless you also change `DATA_FILE` in `covid_forecasting.py`.

## GitHub

The full dataset is excluded by `.gitignore`.

This README is intentionally kept in the repository so another person can clone the project and know exactly how to obtain the required data.
