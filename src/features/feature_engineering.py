import pandas as pd


def sort_data(df):
    return df.sort_values(["Store", "Date"])


#  1. Lag Features (memory)
def add_lag_features(df):
    df["lag_1"] = df.groupby("Store")["Sales"].shift(1)
    df["lag_7"] = df.groupby("Store")["Sales"].shift(7)
    df["lag_30"] = df.groupby("Store")["Sales"].shift(30)
    return df


#  2. Rolling Features (trend)
def add_rolling_features(df):
    df["rolling_mean_7"] = df.groupby("Store")["Sales"].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df.groupby("Store")["Sales"].shift(1).rolling(7).std()

    df["rolling_mean_30"] = df.groupby("Store")["Sales"].shift(1).rolling(30).mean()
    return df


#  3. Date-based Business Features
def add_date_features(df):
    df["IsMonthStart"] = (df["Day"] <= 5).astype(int)
    df["IsMonthEnd"] = (df["Day"] >= 25).astype(int)
    return df


#  4. Promo Impact
def add_promo_features(df):
    df["PromoImpact"] = df["Promo"] * df["Customers"]
    return df

#  5. Competition Feature (SMART)
def add_competition_features(df):
    df["CompetitionOpen"] = (
        (df["Year"] - df["CompetitionOpenSinceYear"]) * 12 +
        (df["Month"] - df["CompetitionOpenSinceMonth"])
    )
    df["CompetitionOpen"] = df["CompetitionOpen"].clip(lower=0)
    return df


#  MAIN FUNCTION
def feature_engineering(df):
    print(" Sorting data...")
    df = sort_data(df)

    print(" Adding lag features...")
    df = add_lag_features(df)

    print(" Adding rolling features...")
    df = add_rolling_features(df)

    print(" Adding date features...")
    df = add_date_features(df)

    print(" Adding promo features...")
    df = add_promo_features(df)

    print(" Adding competition features...")
    df = add_competition_features(df)

    print(" Dropping NaN values...")
    df = df.dropna()

    print(" Feature engineering completed!")
    return df