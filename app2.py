import streamlit as st
import pickle
import numpy as np

# Load the saved Logistic Regression model and TF-IDF vectorizer
with open("final_logistic_regression.pkl", "rb") as f:
    svm = pickle.load(f)

with open("final_tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Set up Streamlit page
st.set_page_config(page_title="CineSentiment", page_icon="🎬", layout="centered")

# Custom CSS for gradient background & styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #a2d2ff, #d3f8e2, #fef9d7);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #104E8B;
        transform: scale(1.05);
    }
    h1, h3 {
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title
st.markdown(
    """
    <h1 style='color: #1E90FF;'>CineSentiment 🎬</h1>
    <h3>Movie Review Sentiment Analysis (Positive / Negative / Neutral)</h3>
    """, unsafe_allow_html=True
)

# Text input
review = st.text_area("✍️ Enter your movie review here:", height=150)

# Predict button
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        # Transform review using TF-IDF
        review_tfidf = tfidf.transform([review])

        # Get probabilities (works with Logistic Regression)
        if hasattr(ls, "predict_proba"):
            probs = ls.predict_proba(review_tfidf)[0]
            prob_neg, prob_pos = probs
        else:
            # Fallback if model doesn’t support predict_proba
            decision = ls.decision_function(review_tfidf)
            prob_pos = 1 / (1 + np.exp(-decision[0]))
            prob_neg = 1 - prob_pos

        # Apply thresholds
        if prob_pos >= 0.6:
            st.success("😘 Sentiment: Positive")
        elif prob_neg >= 0.6:
            st.error("😒 Sentiment: Negative")
        else:
            st.info("😐 Sentiment: Neutral")

