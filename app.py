import re
import joblib
import streamlit as st

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️")

# Load the saved model and TF-IDF
@st.cache_resource
def load_files():
    model = joblib.load("models/lr_model.pkl")
    tfidf = joblib.load("models/tfidf.pkl")
    return model, tfidf

model, tfidf = load_files()

# Same cleaning as in Step 3
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("🕵️ Fake Job Posting Detector")
st.write("Paste a job posting below and the model will tell you if it looks real or fake.")

title = st.text_input("Job title")
description = st.text_area("Job description, requirements and benefits", height=250)

if st.button("Check this job"):
    full_text = title + " " + description

    if len(full_text.strip()) < 20:
        st.warning("Please enter a longer job posting.")
    else:
        cleaned = clean_text(full_text)
        vector = tfidf.transform([cleaned])
        fake_prob = model.predict_proba(vector)[0][1]

        st.subheader("Result")
        if fake_prob >= 0.5:
            st.error(f"⚠️ This looks like a FAKE job posting ({fake_prob * 100:.1f}% fake probability)")
        else:
            st.success(f"✅ This looks like a REAL job posting ({(1 - fake_prob) * 100:.1f}% real probability)")

        st.progress(float(fake_prob))
        st.caption("The bar shows the fake probability. This is a student ML project, so don't treat it as final proof.")
        