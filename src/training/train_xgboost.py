import pandas as pd
import numpy as np
import os
import time

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# PATHS

DATA_PATH = "data/processed/final_featured_train.csv"

MODEL_PATH = "outputs/models/xgboost_model.pkl"
PRED_PATH = "outputs/predictions/xgb_preds.csv"
METRIC_PATH = "outputs/metrics/xgb_metrics.txt"


# LOAD DATA
print(" Loading data...")
df = pd.read_csv(DATA_PATH)
# PREPROCESS

print(" Sorting by Date...")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")


# TIME SPLIT

split = int(len(df) * 0.8)

train_data = df[:split]
test_data = df[split:]

X_train = train_data.drop(["Sales", "Date"], axis=1)
y_train = train_data["Sales"]

X_test = test_data.drop(["Sales", "Date"], axis=1)
y_test = test_data["Sales"]

#  REMOVE OBJECT COLUMNS
X_train = X_train.select_dtypes(exclude=["object"])
X_test = X_test.select_dtypes(exclude=["object"])

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")


# MODEL

print(" Training XGBoost...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

train_time = end_time - start_time


# PREDICT

print(" Predicting...")

y_pred = model.predict(X_test)


# METRICS

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"Training Time: {train_time:.2f} sec")


# SAVE OUTPUTS

print(" Saving outputs...")

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

# Save model
import joblib
joblib.dump(model, MODEL_PATH)

# Save predictions
pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
pred_df.to_csv(PRED_PATH, index=False)

# Save metrics
with open(METRIC_PATH, "w") as f:
    f.write(f"RMSE: {rmse}\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"MAPE: {mape}\n")
    f.write(f"Training Time: {train_time:.2f} sec\n")

print("XGBoost Training Completed!")

from src.evaluation.metrics import save_results

save_results(
    model_name="XGBoost",
    model=model,
    y_test=y_test,
    y_pred=y_pred,
    rmse=rmse,
    mae=mae,
    mape=mape,
    time_taken=end_time - start_time
)