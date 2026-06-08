
#  LOAD DATA
print(" Loading data...")

df = pd.read_csv("data/processed/featured_data.csv")


#  TIME SORT (DO THIS FIRST)

df = df.sort_values("Date")

#  TARGET & FEATURES

y = df["Sales"]
X = df.drop(["Sales"], axis=1)

#  TRAIN TEST SPLIT (80-20)

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]