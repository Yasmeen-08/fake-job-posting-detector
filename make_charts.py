import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

os.makedirs("charts", exist_ok=True)

# 1. Load data and models, same split as before
df = pd.read_csv("data/cleaned_jobs.csv")
df["text"] = df["text"].fillna("")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["fraudulent"],
    test_size=0.2, random_state=42, stratify=df["fraudulent"]
)

tfidf = joblib.load("models/tfidf.pkl")
lr_model = joblib.load("models/lr_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
X_test_tfidf = tfidf.transform(X_test)

# ---------- Chart 1: Real vs Fake count ----------
plt.figure(figsize=(5, 4))
ax = sns.countplot(x=df["fraudulent"].map({0: "Real", 1: "Fake"}),
                   order=["Real", "Fake"], palette=["#4c9f70", "#d9534f"])
ax.bar_label(ax.containers[0])
plt.title("Real vs Fake Job Postings")
plt.xlabel("")
plt.ylabel("Number of postings")
plt.tight_layout()
plt.savefig("charts/1_class_balance.png", dpi=150)
plt.close()

# ---------- Chart 2: Confusion matrices ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, model) in zip(axes, [("Logistic Regression", lr_model),
                                    ("Random Forest", rf_model)]):
    cm = confusion_matrix(y_test, model.predict(X_test_tfidf))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("charts/2_confusion_matrices.png", dpi=150)
plt.close()

# ---------- Chart 3: Model comparison ----------
results = pd.read_csv("models/model_comparison.csv", index_col=0)
results.plot(kind="bar", figsize=(8, 4.5), rot=0, color=["#3b7dd8", "#f0a030"])
plt.title("Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("charts/3_model_comparison.png", dpi=150)
plt.close()

# ---------- Chart 4: Top words that signal a FAKE job ----------
words = tfidf.get_feature_names_out()
coefs = lr_model.coef_[0]
top = pd.Series(coefs, index=words).sort_values(ascending=False).head(15)

plt.figure(figsize=(7, 5))
top.sort_values().plot(kind="barh", color="#d9534f")
plt.title("Top 15 Words Pointing to a Fake Job")
plt.xlabel("Model weight")
plt.tight_layout()
plt.savefig("charts/4_top_fake_words.png", dpi=150)
plt.close()

print("Done! 4 charts saved in the charts folder.")