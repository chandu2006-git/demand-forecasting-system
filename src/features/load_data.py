import pandas as pd


def load_train_data(path="data/Rossmann Store Sales Dataset/train.csv"):
    return pd.read_csv(path)


def load_store_data(path="data/Rossmann Store Sales Dataset/store.csv"):
    return pd.read_csv(path)


def load_test_data(path="data/Rossmann Store Sales Dataset/test.csv"):
    return pd.read_csv(path)


def load_all_data():
    print("📥 Loading all datasets...")

    train = load_train_data()
    store = load_store_data()
    test = load_test_data()

    print("✅ Data loaded successfully!")
    print(f"Train shape: {train.shape}")
    print(f"Store shape: {store.shape}")
    print(f"Test shape: {test.shape}")

    return train, store, test