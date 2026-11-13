import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load data and make the SAME split as before
df = pd.read_csv("data/cleaned_jobs.csv")
df["text"] = df["text"].fillna("")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["fraudulent"],
    test_size=0.2, random_state=42, stratify=df["fraudulent"]
)

# 2. Load the saved TF-IDF and Logistic Regression
tfidf = joblib.load("models/tfidf.pkl")
lr_model = joblib.load("models/lr_model.pkl")

X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 3. Train Random Forest (this takes a minute or two)
print("Training Random Forest... please wait")
rf_model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_model.fit(X_train_tfidf, y_train)

# 4. Compare both models on Fake jobs
def score(model):
    preds = model.predict(X_test_tfidf)
    return [
        round(accuracy_score(y_test, preds), 4),
        round(precision_score(y_test, preds), 4),
        round(recall_score(y_test, preds), 4),
        round(f1_score(y_test, preds), 4),
    ]

results = pd.DataFrame(
    {
        "Logistic Regression": score(lr_model),
        "Random Forest": score(rf_model),
    },
    index=["Accuracy", "Fake Precision", "Fake Recall", "Fake F1"],
)

print()
print(results)

# 5. Save the Random Forest and the comparison table
joblib.dump(rf_model, "models/rf_model.pkl")
results.to_csv("models/model_comparison.csv")
print()
print("Saved rf_model.pkl and model_comparison.csv in the models folder.")