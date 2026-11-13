import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load cleaned data
df = pd.read_csv("data/cleaned_jobs.csv")
df["text"] = df["text"].fillna("")

# 2. Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["fraudulent"],
    test_size=0.2, random_state=42, stratify=df["fraudulent"]
)

# 3. TF-IDF: turn text into numbers
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. Train Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

# 5. Test it
preds = model.predict(X_test_tfidf)

print("Accuracy:", round(accuracy_score(y_test, preds), 4))
print()
print("Classification report:")
print(classification_report(y_test, preds, target_names=["Real", "Fake"]))
print("Confusion matrix:")
print(confusion_matrix(y_test, preds))

# 6. Save the model and vectorizer
joblib.dump(model, "models/lr_model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
print()
print("Model saved in the models folder.")
