import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.data.load_data import load_all_data


def merge_data(train, test, store):
    train = train.merge(store, on="Store", how="left")
    test = test.merge(store, on="Store", how="left")
    return train, test


def handle_missing_values(df):
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    df["CompetitionOpenSinceMonth"].fillna(0, inplace=True)
    df["CompetitionOpenSinceYear"].fillna(0, inplace=True)

    df["Promo2SinceWeek"].fillna(0, inplace=True)
    df["Promo2SinceYear"].fillna(0, inplace=True)
    df["PromoInterval"].fillna(0, inplace=True)

    return df


def create_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    return df


def clean_data(df):
    if "Sales" in df.columns:
        df = df[df["Open"] != 0]
        df = df[df["Sales"] > 0]
    return df


def encode_categorical(df):
    le = LabelEncoder()

    for col in ["StoreType", "Assortment", "StateHoliday"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def preprocess():
    print(" Loading data...")
    train, store, test = load_all_data()

    print(" Merging datasets...")
    train, test = merge_data(train, test, store)

    print(" Handling missing values...")
    train = handle_missing_values(train)
    test = handle_missing_values(test)

    print(" Creating date features...")
    train = create_date_features(train)
    test = create_date_features(test)

    print(" Cleaning training data...")
    train = clean_data(train)

    print(" Encoding categorical features...")
    train = encode_categorical(train)
    test = encode_categorical(test)

    print(" Saving processed data...")
    train.to_csv("data/processed/final_train.csv", index=False)
    test.to_csv("data/processed/final_test.csv", index=False)

    print(" Preprocessing completed successfully!")


if __name__ == "__main__":
    preprocess()