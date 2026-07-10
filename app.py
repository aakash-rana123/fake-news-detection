# import streamlit as st
# import joblib
#
# # --------------------------
# # Load model and vectoriser once (cached)
# # --------------------------
# @st.cache_resource
# def load_model_and_vectorizer():
#     tfidf = joblib.load("tfidf_vectorizer.pkl")
#     pac_model = joblib.load("pac_model.pkl")
#     return tfidf, pac_model
#
# tfidf, pac_model = load_model_and_vectorizer()
#
# # --------------------------
# # Streamlit UI
# # --------------------------
# st.title("Fake News Detection Web App")
#
# st.write(
#     "Enter a news headline and article text below. "
#     "The app will classify it as Fake or Real based on a trained NLP model."
# )
#
# # User input fields
# title_input = st.text_input("News Title")
# text_input = st.text_area("News Content")
#
# # Classification button
# if st.button("Classify"):
#     if not title_input and not text_input:
#         st.warning("Please enter at least a title or some content.")
#     else:
#         # Combine title and content like in training
#         full_text = title_input + " " + text_input
#
#         # Transform with TF-IDF
#         X_new = tfidf.transform([full_text])
#
#         # Predict with PassiveAggressive model
#         pred = pac_model.predict(X_new)[0]
#
#         label = "Real News" if pred == 1 else "Fake News"
#
#         st.subheader(f"Prediction: {label}")


import streamlit as st
import joblib

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# --------------------------
# Load model once (cached)
# --------------------------
@st.cache_resource
def load_model_and_vectorizer():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    pac_model = joblib.load("pac_model.pkl")
    return tfidf, pac_model

tfidf, pac_model = load_model_and_vectorizer()

# --------------------------
# App Header
# --------------------------
st.title("📰 Fake News Detection System")
st.markdown("**MSc Data Science Dissertation Project – University of Essex**")
st.markdown("---")

st.write(
    "Enter a news headline and article text below. "
    "The system will classify it as **Real** or **Fake** using a "
    "PassiveAggressive Classifier trained on ~45,000 news articles."
)

# --------------------------
# User Input
# --------------------------
st.subheader("Enter News Article")
title_input = st.text_input("News Title", placeholder="e.g. UK economy grows by 0.7 percent...")
text_input = st.text_area("News Content", placeholder="Paste the article text here...", height=200)

# --------------------------
# Classify Button
# --------------------------
if st.button("🔍 Classify Article", use_container_width=True):

    if not title_input and not text_input:
        st.warning("Please enter at least a title or some article content.")

    else:
        # Combine and transform
        full_text = title_input + " " + text_input
        X_new = tfidf.transform([full_text])
        pred = pac_model.predict(X_new)[0]

        # Get confidence score
        decision = pac_model.decision_function(X_new)[0]

        # Normalize to 0-100% confidence (rough estimate)
        import numpy as np
        confidence = round(min(abs(decision) / 3 * 100, 99), 1)

        st.markdown("---")
        st.subheader("Result")

        if pred == 1:
            st.success(f"✅ REAL NEWS  —  Confidence: {confidence}%")
            st.markdown(
                "<div style='background-color:#d4edda; padding:20px; "
                "border-radius:10px; text-align:center;'>"
                "<h2 style='color:#155724;'>✅ Real News</h2>"
                f"<p style='color:#155724; font-size:18px;'>Confidence: {confidence}%</p>"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.error(f"🚨 FAKE NEWS  —  Confidence: {confidence}%")
            st.markdown(
                "<div style='background-color:#f8d7da; padding:20px; "
                "border-radius:10px; text-align:center;'>"
                "<h2 style='color:#721c24;'>🚨 Fake News</h2>"
                f"<p style='color:#721c24; font-size:18px;'>Confidence: {confidence}%</p>"
                "</div>",
                unsafe_allow_html=True
            )

        # Show input summary
        st.markdown("---")
        st.subheader("Input Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Title Length", f"{len(title_input.split())} words")
        with col2:
            st.metric("Content Length", f"{len(text_input.split())} words")

# --------------------------
# Disclaimer
# --------------------------
st.markdown("---")
st.markdown(
    """
    <div style='background-color:#fff3cd; padding:15px; border-radius:8px;'>
    <b>⚠️ Disclaimer:</b> This system was trained on a specific Kaggle dataset 
    (2016–2017 US political news). Performance on recent or non-US news may vary. 
    This tool is for academic research purposes only and should not be used as 
    the sole basis for judging the credibility of any news article.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.caption("Built with Python, scikit-learn, and Streamlit | University of Essex MSc Data Science 2026")