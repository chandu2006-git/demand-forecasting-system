import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib


def save_results(model_name, model, y_test, y_pred, rmse, mae, mape, time_taken):
    
    # Create folders
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

   
    # 1. SAVE METRICS
  
    metrics_df = pd.DataFrame([{
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Time": time_taken
    }])

    file_path = "outputs/metrics/model_metrics.csv"

    if not os.path.exists(file_path):
        metrics_df.to_csv(file_path, index=False)
    else:
        metrics_df.to_csv(file_path, mode='a', header=False, index=False)

    # 2. SAVE PREDICTIONS

    pred_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    pred_df.to_csv(f"outputs/predictions/{model_name.lower()}_preds.csv", index=False)

    # 3. SAVE MODEL

    joblib.dump(model, f"outputs/models/{model_name.lower()}_model.pkl")

    # 4. PLOTS

    # Actual vs Predicted
    plt.figure()
    plt.plot(y_test.values[:200], label="Actual")
    plt.plot(y_pred[:200], label="Predicted")
    plt.legend()
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.savefig(f"outputs/plots/{model_name.lower()}_actual_vs_pred.png")
    plt.close()

    # Error distribution
    errors = y_test.values - y_pred

    plt.figure()
    plt.hist(errors, bins=50)
    plt.title(f"{model_name} - Error Distribution")
    plt.savefig(f"outputs/plots/{model_name.lower()}_error.png")
    plt.close()