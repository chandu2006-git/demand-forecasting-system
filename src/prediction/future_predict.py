import pandas as pd
import numpy as np
import joblib
import os

print(" Script Started...")

# 1. LOAD MODEL
model_path = "outputs/models/xgboost_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f" Model not found at {model_path}")

model = joblib.load(model_path)
print(" Model Loaded")

# 2. LOAD TRAIN DATA
data_path = "data/processed/train_features.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f" Data not found at {data_path}")

df = pd.read_csv(data_path)

if "Date" not in df.columns or "Sales" not in df.columns:
    raise ValueError(" 'Date' or 'Sales' column missing in dataset")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

print(" Data Loaded:", df.shape)


# 3. CREATE FUTURE DATES

future_days = 30

future_dates = pd.date_range(
    start=df["Date"].max() + pd.Timedelta(days=1),
    periods=future_days
)

future_df = pd.DataFrame({"Date": future_dates})

print(" Future dates created")

# 4. DATE FEATURES
future_df["Year"] = future_df["Date"].dt.year
future_df["Month"] = future_df["Date"].dt.month
future_df["Day"] = future_df["Date"].dt.day
future_df["DayOfWeek"] = future_df["Date"].dt.dayofweek


# 5. STATIC FEATURES (ASSUMPTIONS)

future_df["Store"] = 1
future_df["Promo"] = 0
future_df["StateHoliday"] = 0
future_df["SchoolHoliday"] = 0

print(" Basic features added")


# 6. INITIALIZE LAG VALUES

if len(df) < 30:
    raise ValueError(" Not enough data for lag features (need at least 30 rows)")

last_sales = list(df["Sales"].tail(30).values)

predictions = []

print(" Starting recursive predictions...")


# 7. GET MODEL FEATURE ORDER

try:
    feature_cols = model.get_booster().feature_names
except:
    feature_cols = df.drop(columns=["Sales", "Date"]).columns.tolist()

print(" Feature columns loaded")

# 8. ITERATIVE PREDICTION

for i in range(future_days):

    row = future_df.iloc[i].copy()

    # Lag features
    row["lag_1"] = last_sales[-1]
    row["lag_7"] = last_sales[-7]

    # Rolling features
    row["rolling_mean_7"] = np.mean(last_sales[-7:])
    row["rolling_std_7"] = np.std(last_sales[-7:])

    # Convert to DataFrame
    row_df = pd.DataFrame([row])

    # Ensure all columns exist
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0

    # Match exact order
    row_df = row_df[feature_cols]

    # Predict
    pred = model.predict(row_df)[0]
    predictions.append(pred)

    # Update lag values
    last_sales.append(pred)

    if i % 5 == 0:
        print(f" Day {i+1} predicted")

print(" Predictions completed:", len(predictions))


# 9. SAVE OUTPUT

output_dir = "outputs/predictions"
os.makedirs(output_dir, exist_ok=True)

future_df["Predicted_Sales"] = predictions

output_path = os.path.join(output_dir, "future_predictions.csv")
future_df.to_csv(output_path, index=False)

print(" File saved at:", output_path)
print(" Future predictions completed successfully!")