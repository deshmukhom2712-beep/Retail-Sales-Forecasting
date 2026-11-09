# Retail Sales Analytics & Demand Forecasting

## Project Overview

A collaborative Data Science project that analyzes historical retail sales and forecasts daily store-level sales using Machine Learning, built on the Kaggle [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset.

## Technologies

* Python
* MySQL
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* XGBoost
* SHAP
* Streamlit

## Project Objectives

* Analyze historical retail sales.
* Perform SQL-based business analysis.
* Identify sales trends and patterns.
* Compare Random Forest and XGBoost for store-level daily sales forecasting.
* Explain model predictions using SHAP.
* Develop an interactive Streamlit dashboard.

## Dataset

* Rossmann Store Sales (Kaggle): daily sales for 1,115 stores from 2013-01-01 to 2015-07-31, with promotions, holidays, store type, assortment and competition details. Download `train.csv`, `test.csv` and `store.csv` into `data/raw/` (not included in this repo).

## Methodology

* Removed closed-store and zero-sales days, then merged `train` with `store`
* Chronological split: train 2013-01-01 to 2015-06-19, test the final 6 weeks
* Compared Random Forest and XGBoost against a train-mean baseline

## Results (baseline features)

| Model | MAE | RMSE | R² | RMSPE |
|---|---:|---:|---:|---:|
| Baseline (train mean) | 2252.96 | n/a | n/a | n/a |
| Random Forest | 1232.87 | 1738.58 | 0.6761 | 0.2847 |
| XGBoost | 944.68 | 1300.80 | 0.8187 | 0.2256 |

*Evaluated on the final 6 weeks (2015-06-20 to 2015-07-31) as a held-out test set.*


## Key Insights
_To be added after EDA is finalized._

## Project Structure
```
├── data/raw/        # Kaggle CSVs (gitignored)
├── notebooks/       # data understanding, EDA, modeling
├── sql/             # schema and business queries
├── app/             # Streamlit dashboard
└── README.md
```


## Team

* Om Deshmukh : Data cleaning, EDA, machine learning, evaluation
* Anish Thorat : MySQL database design and SQL analysis

## Project Status
🚧 In Development
