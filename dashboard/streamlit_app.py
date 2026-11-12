from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "data" / "raw" / "train.csv"
PREDICTIONS = ROOT / "data" / "test_predictions.parquet"
FIGURES = ROOT / "reports" / "figures"

DAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

st.set_page_config(page_title="Retail Sales Dashboard", page_icon="📈", layout="wide")


@st.cache_data
def load_train():
    df = pd.read_csv(
        TRAIN_CSV,
        usecols=["Store", "DayOfWeek", "Date", "Sales", "Open", "Promo"],
        parse_dates=["Date"],
    )
    return df[(df["Open"] == 1) & (df["Sales"] > 0)]


@st.cache_data
def load_predictions():
    return pd.read_parquet(PREDICTIONS)


def page_overview():
    st.title("Sales Overview")
    if not TRAIN_CSV.exists():
        st.error("data/raw/train.csv not found. Download the Rossmann dataset from Kaggle into data/raw/.")
        return

    df = load_train()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sales", f"{df['Sales'].sum() / 1e9:.2f} B")
    c2.metric("Avg daily sales per store", f"{df['Sales'].mean():,.0f}")
    c3.metric("Stores", f"{df['Store'].nunique():,}")
    c4.metric("Period", f"{df['Date'].min():%b %Y} to {df['Date'].max():%b %Y}")

    monthly = (
        df.set_index("Date")["Sales"].resample("MS").mean().reset_index(name="Avg daily sales")
    )
    st.plotly_chart(
        px.line(monthly, x="Date", y="Avg daily sales", title="Monthly average daily sales per store"),
        use_container_width=True,
    )

    left, right = st.columns(2)
    weekday = df.groupby("DayOfWeek")["Sales"].mean().rename(index=DAY_NAMES).reset_index()
    left.plotly_chart(
        px.bar(weekday, x="DayOfWeek", y="Sales", title="Average sales by weekday"),
        use_container_width=True,
    )
    promo = df.groupby("Promo")["Sales"].mean().rename(index={0: "No promo", 1: "Promo"}).reset_index()
    right.plotly_chart(
        px.bar(promo, x="Promo", y="Sales", title="Average sales: promo vs no promo"),
        use_container_width=True,
    )


def page_store_explorer():
    st.title("Store Explorer: Actual vs Predicted")
    if not PREDICTIONS.exists():
        st.error("data/test_predictions.parquet not found. Run the modeling notebook to generate it.")
        return

    preds = load_predictions()
    store_id = st.selectbox("Select a store", sorted(preds["Store"].unique()))
    sdf = preds[preds["Store"] == store_id].sort_values("Date")

    mae = (sdf["Sales"] - sdf["Predicted"]).abs().mean()
    mape = ((sdf["Sales"] - sdf["Predicted"]).abs() / sdf["Sales"]).mean() * 100
    c1, c2 = st.columns(2)
    c1.metric("MAE (this store)", f"{mae:,.0f}")
    c2.metric("MAPE (this store)", f"{mape:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sdf["Date"], y=sdf["Sales"], name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=sdf["Date"], y=sdf["Predicted"], name="Predicted", mode="lines+markers"))
    fig.update_layout(title=f"Store {store_id}: last 6 weeks (test period)", xaxis_title="Date", yaxis_title="Sales")
    st.plotly_chart(fig, use_container_width=True)


def page_model():
    st.title("Model Results and Insights")
    results = pd.DataFrame(
        {
            "Model": [
                "Baseline (train mean)",
                "Random Forest",
                "XGBoost",
                "XGBoost + engineered features",
            ],
            "MAE": [2252.96, 1232.87, 944.68, 646.97],
            "RMSE": [None, 1738.58, 1300.80, 945.74],
            "R2": [None, 0.6761, 0.8187, 0.9041],
            "RMSPE": [None, 0.2847, 0.2256, 0.1356],
        }
    )
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.caption("Evaluated on the final 6 weeks (2015-06-20 to 2015-07-31) as a held-out test set.")

    st.subheader("What drives the forecasts (SHAP)")
    left, right = st.columns(2)
    bar, summary = FIGURES / "shap_bar.png", FIGURES / "shap_summary.png"
    if bar.exists():
        left.image(str(bar), caption="Mean absolute SHAP value per feature")
    if summary.exists():
        right.image(str(summary), caption="SHAP summary plot")
    if not (bar.exists() or summary.exists()):
        st.info("SHAP plots not found in reports/figures/. Run the SHAP notebook first.")
    st.caption("SHAP values are in log-sales units, computed on a 2,000-row test sample.")


PAGES = {
    "Overview": page_overview,
    "Store Explorer": page_store_explorer,
    "Model & Insights": page_model,
}

st.sidebar.title("Retail Sales Forecasting")
choice = st.sidebar.radio("Go to", list(PAGES))
PAGES[choice]()
