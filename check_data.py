import pandas as pd

df = pd.read_csv("data/fake_job_postings.csv")

print("Rows and columns:", df.shape)
print()
print("Column names:")
print(df.columns.tolist())
print()
print("Real (0) vs Fake (1):")
print(df["fraudulent"].value_counts())
