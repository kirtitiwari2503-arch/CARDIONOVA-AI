import pandas as pd

print("Feature Engineering Started")

df = pd.read_csv("data/cardio_train.csv", sep=";")

df["age_years"] = (df["age"] / 365).round(1)

df["BMI"] = (
    df["weight"] /
    ((df["height"] / 100) ** 2)
).round(2)

df["obesity"] = (df["BMI"] >= 30).astype(int)

print(df[["age_years", "BMI", "obesity"]].head())

df.to_csv("data/cardio_processed.csv", index=False)

print("Processed dataset saved!")