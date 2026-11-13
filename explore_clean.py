import pandas as pd

df = pd.read_csv("data/fake_job_postings.csv")

# ---------- EXPLORE ----------
print("Missing values per column:")
print(df.isnull().sum())
print()

print("Fake rate (0 to 1) for these columns:")
for col in ["telecommuting", "has_company_logo", "has_questions"]:
    print(df.groupby(col)["fraudulent"].mean().round(3))
    print()

# ---------- CLEAN ----------
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
df[text_cols] = df[text_cols].fillna("")

df["text"] = (
    df["title"] + " " + df["company_profile"] + " " +
    df["description"] + " " + df["requirements"] + " " + df["benefits"]
)

df["text"] = df["text"].str.lower()
df["text"] = df["text"].str.replace(r"[^a-z\s]", " ", regex=True)
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

df = df[["text", "fraudulent"]]
df.to_csv("data/cleaned_jobs.csv", index=False)

print("Cleaned data saved. Shape:", df.shape)
print()
print("Sample text:")
print(df["text"].iloc[0][:300])
