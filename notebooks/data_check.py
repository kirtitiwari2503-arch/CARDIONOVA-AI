import pandas as pd

df = pd.read_csv("data/cardio_train.csv", sep=";")

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(list(df.columns))

print("\nMissing Values:")
print(df.isnull().sum())