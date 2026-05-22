import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Demand Forecast", layout="wide")
st.title("📊 Advanced Demand Forecasting System")

# ================================
# LOAD MODEL & DATA
# ================================
@st.cache_resource
def load_model():
    return joblib.load("outputs/models/xgboost_model.pkl")
@st.cache_data
def load_data():
    df = pd.read_csv("data/sample_data.csv", encoding='latin1')

    # 🔥 Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Rename properly
    df = df.rename(columns={
        "store": "Store",
        "sales": "Sales",
        "date": "Date"
    })

    # Date fix
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])

    return df.sort_values("Date")

model = load_model()
df = load_data()

st.write("Columns in dataset:", df.columns)
# ================================
# INTELLIGENCE LAYER
# ================================
def generate_insights(df, promo, holidays):

    insights = []

    avg_sales = df["Predicted Sales"].mean()
    max_sales = df["Predicted Sales"].max()
    min_sales = df["Predicted Sales"].min()

    # Trend
    if df["Predicted Sales"].iloc[-1] > df["Predicted Sales"].iloc[0]:
        insights.append("📈 Sales are expected to increase over time.")
    else:
        insights.append("📉 Sales show a decreasing trend.")

    # Volatility
    if (max_sales - min_sales) > 500:
        insights.append("⚠️ High fluctuation detected in sales.")

    # Promo
    if promo == 1:
        insights.append("🎯 Promotional activity may boost sales.")

    # Holidays
    if holidays == 1:
        insights.append("🏖️ Holidays may influence customer behavior.")

    # Weekend pattern
    weekend_days = df["Date"].apply(lambda x: x.weekday() >= 5)
    if weekend_days.sum() > 0:
        insights.append("🗓️ Weekend sales variation expected.")

    return insights

# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.header("⚙️ Input Parameters")

store = st.sidebar.number_input("🏪 Store ID", min_value=1, value=1)

start_date = st.sidebar.date_input("📅 Start Date")
end_date = st.sidebar.date_input("📅 End Date")

promo = st.sidebar.selectbox("🎯 Promo", [0, 1])
stateHoliday = st.sidebar.selectbox("🏖️ State Holiday", [0, 1])
schoolHoliday = st.sidebar.selectbox("🎒 School Holiday", [0, 1])

predict_btn = st.sidebar.button("🚀 Predict")

# ================================
# MAIN LOGIC
# ================================
if predict_btn:

    if start_date >= end_date:
        st.error("❌ End date must be after start date")

    else:
        st.info("🔄 Generating predictions...")

        future_dates = pd.date_range(start=start_date, end=end_date)
        days = len(future_dates)

        # 🔥 FIX 4: Safe column access
        if "Store" not in df.columns:
            st.error("❌ 'Store' column not found in dataset")
            st.stop()

        store_data = df[df["Store"] == store]

        if len(store_data) < 30:
            st.error("❌ Not enough historical data for this store")

        else:
            last_sales = list(store_data["Sales"].tail(30).values)
            predictions = []

            try:
                feature_cols = model.get_booster().feature_names
            except:
                feature_cols = df.drop(columns=["Sales", "Date"]).columns.tolist()

            for i in range(days):

                date = future_dates[i]

                row = {
                    "Store": store,
                    "Promo": promo,
                    "StateHoliday": stateHoliday,
                    "SchoolHoliday": schoolHoliday,
                    "Year": date.year,
                    "Month": date.month,
                    "Day": date.day,
                    "DayOfWeek": date.dayofweek,

                    # Lag Features
                    "lag_1": last_sales[-1],
                    "lag_7": last_sales[-7],
                    "lag_30": last_sales[-30],

                    # Rolling Features
                    "rolling_mean_7": np.mean(last_sales[-7:]),
                    "rolling_mean_30": np.mean(last_sales[-30:]),
                    "rolling_std_7": np.std(last_sales[-7:])
                }

                row_df = pd.DataFrame([row])

                for col in feature_cols:
                    if col not in row_df.columns:
                        row_df[col] = 0

                row_df = row_df[feature_cols]

                pred = model.predict(row_df)[0]

                predictions.append({
                    "Date": date.date(),
                    "Predicted Sales": round(pred, 2)
                })

                last_sales.append(pred)

            result_df = pd.DataFrame(predictions)

            st.success("✅ Prediction Complete!")

            # ================================
            # KPI METRICS
            # ================================
            col1, col2, col3 = st.columns(3)

            col1.metric("📈 Total Sales", int(result_df["Predicted Sales"].sum()))
            col2.metric("📊 Avg Sales", int(result_df["Predicted Sales"].mean()))
            col3.metric("🔥 Max Sales", int(result_df["Predicted Sales"].max()))

            st.divider()

            # ================================
            # AI INSIGHTS
            # ================================
            insights = generate_insights(
                result_df,
                promo,
                stateHoliday or schoolHoliday
            )

            st.subheader("🧠 AI Insights")
            for insight in insights:
                st.write(insight)

            st.divider()

            # ================================
            # GRAPH + TABLE
            # ================================
            colA, colB = st.columns([2, 1])

            with colA:
                st.subheader("📈 Sales Forecast")
                st.line_chart(result_df.set_index("Date"))

            with colB:
                st.subheader("📋 Data")
                st.dataframe(result_df)

            st.divider()

            # ================================
            # DOWNLOAD
            # ================================
            st.download_button(
                "📥 Download CSV",
                result_df.to_csv(index=False),
                file_name="forecast.csv",
                mime="text/csv"
            )