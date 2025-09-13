import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Load Data & Model
# -----------------------
@st.cache_resource
def load_data():
    # Load combined dataset
    df = pd.read_csv("combined_prompts.csv")
    
    # Load vectorizer
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    return df, vectorizer

df, vectorizer = load_data()
prompts = df["prompt"].tolist()

# -----------------------
# Suggestion Function
# -----------------------
def suggest_prompt(user_input, top_n=3):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, vectorizer.transform(prompts)).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(prompts[i], similarities[i]) for i in top_indices]

# -----------------------
# Streamlit UI
# -----------------------
# Page Config
st.set_page_config(
    page_title="Prompt Rewriter",
    page_icon="✨",
    layout="centered",
)

# Custom CSS for fonts and colors
st.markdown(
    """
    <style>
    body {
        background-color: #fdfcfb;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .result {
        background-color: #f0f0f0;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        font-size: 16px;
        color: #2c3e50;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown("<div class='title'>✨ Prompt Rewriter ✨</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Transform your messy prompts into polished, engineered prompts</div>", unsafe_allow_html=True)

# User Input
user_input = st.text_area("✍️ Enter your prompt:", placeholder="e.g. help me cook dinner I am vegetarian")

# Submit Button
if st.button("Rewrite Prompt 🚀"):
    if user_input.strip():
        results = suggest_prompt(user_input, top_n=3)
        st.markdown("### 🔮 Suggested Prompts:")
        for prompt, score in results:
            st.markdown(f"<div class='result'><b>Score:</b> {score:.2f}<br>{prompt}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a prompt to rewrite.")