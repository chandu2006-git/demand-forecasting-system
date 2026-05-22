
#  IMPORTS

import time
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# LOAD DATA

print(" Loading data...")

df = pd.read_csv("data/processed/train_features.csv")  # adjust path if needed


#  TARGET & FEATURES

y = df["Sales"]
X = df.drop(["Sales"], axis=1)
y = df["Sales"]
X = df.drop(["Sales"], axis=1)

#  ADD THIS
X = X.drop(["Date", "PromoInterval"], axis=1)

#  TIME SORT (IMPORTANT)

df = df.sort_values("Date")


#  TRAIN TEST SPLIT (80-20)

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")


# ⏱️ START TIMER

start_time = time.time()

#  MODEL

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)


#  TRAIN

model.fit(X_train, y_train)


#  PREDICT

y_pred = model.predict(X_test)


#  METRICS

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100


# END TIMER

end_time = time.time()
#  RESULTS

print("\n LightGBM Results")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"Time : {end_time - start_time:.2f} sec")

from src.evaluation.metrics import save_results

save_results(
    model_name="LightGBM",
    model=model,
    y_test=y_test,
    y_pred=y_pred,
    rmse=rmse,
    mae=mae,
    mape=mape,
    time_taken=end_time - start_time
)