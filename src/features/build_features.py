import pandas as pd
import os
from src.features.feature_engineering import feature_engineering

print("📥 Loading processed data...")
train = pd.read_csv("data/processed/final_train.csv")

print("⚙️ Applying feature engineering...")
train = feature_engineering(train)

print("📊 Final shape:", train.shape)

os.makedirs("data/processed", exist_ok=True)

print("💾 Saving featured data...")
train.to_csv("data/processed/train_features.csv", index=False)

print("📁 Saved at: data/processed/train_features.csv")