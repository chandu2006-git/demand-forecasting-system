import pandas as pd
import numpy as np
import time

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. LOAD DATA (STRICT RULE)

print(" Loading feature dataset...")

df = pd.read_csv("data/processed/train_features.csv")


# 2. BASIC CLEANING

print(" Cleaning data...")

# Drop Date (NOT needed for model)
if "Date" in df.columns:
    df = df.drop("Date", axis=1)


# 3. SORT FOR TIME SERIES

print(" Sorting data...")

df = df.sort_values(by=df.columns[0])  # already sorted earlier, safe step


# 4. SPLIT (TIME SERIES)

print(" Splitting data...")

X = df.drop("Sales", axis=1)
y = df["Sales"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# 5. HANDLE CATEGORICAL FEATURES

print(" Detecting categorical features...")

cat_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Categorical columns:", cat_features)



# 6. MODEL TRAINING

print(" Training CatBoost...")

start_time = time.time()

model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=100
)

model.fit(X_train, y_train, cat_features=cat_features)

end_time = time.time()

# 7. PREDICTIONS

print(" Predicting...")

y_pred = model.predict(X_test)


# 8. METRICS

print(" Calculating metrics...")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100


# 9. RESULTS

print("\n CATBOOST RESULTS")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"MAPE : {mape:.2f}%")

from src.evaluation.metrics import save_results

save_results(
    model_name="CatBoost",
    model=model,
    y_test=y_test,
    y_pred=y_pred,
    rmse=rmse,
    mae=mae,

    mape=mape,
    time_taken=end_time - start_time
)
print(" Saving results...")