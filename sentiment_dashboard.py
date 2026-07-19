import streamlit as st
import pandas as pd
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

from sentiment_batch import predict_dataframe

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

@st.cache_resource
def load_models():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("📊 Social Media Sentiment Analysis Dashboard")

try:
    model, vectorizer = load_models()
    user_input = st.text_area("Enter text to analyze:", height=100)
    if st.button("Analyze"):
        if user_input:
            cleaned = preprocess_text(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            if pred == 'positive':
                st.success("😊 Sentiment: Positive")
            elif pred == 'negative':
                st.error("😠 Sentiment: Negative")
            else:
                st.info("😐 Sentiment: Neutral")

    st.markdown("### Batch CSV Analysis")
    uploaded_file = st.file_uploader(
        "Upload CSV with a text, tweet, full_text, content, comment, review, message, or caption column",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        try:
            results = predict_dataframe(
                df,
                model=model,
                vectorizer=vectorizer,
                preprocess=preprocess_text,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success(f"Analyzed {len(results)} rows")
            st.dataframe(results.head())
            st.download_button(
                "Download Predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_predictions.csv",
                mime="text/csv",
            )
except Exception as e:
    st.error(f"Error: {e}")
