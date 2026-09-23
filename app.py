import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


# --------------------------------------------------
# Download NLTK Resources
# --------------------------------------------------

@st.cache_resource
def download_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)


download_nltk()

# --------------------------------------------------
# Load Model and TF-IDF Vectorizer
# --------------------------------------------------

@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as file:
        bundle = pickle.load(file)

    return bundle["model"], bundle["vectorizer"]

model, vectorizer = load_model()

# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------


def clean_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(r'https?://\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'#(\w+)', r'\1', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_text(text):

    text = clean_text(text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))

    tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    lemmatizer = WordNetLemmatizer()

    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]

    return " ".join(tokens)
# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 35px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 25px;
        border: 1px solid #dddddd;
    }

    .sentiment {
        font-size: 32px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">💬 Sentiment Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze whether a text expresses a Positive negative or  Neutral sentiment'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Input
# --------------------------------------------------

st.subheader("Enter your text")

text = st.text_area(
    "Write a sentence, review, or tweet:",
    placeholder="Example: I really enjoyed this product!",
    height=150
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Preprocess text
        processed_text = preprocess_text(text)

        # Convert text to TF-IDF
        text_vector = vectorizer.transform([processed_text])

        # Prediction
        prediction = model.predict(text_vector)[0]

        # Your notebook uses:
        # neutral = 2
        # positive = 4
        
        

        if prediction == 4:
            sentiment = "Positive 😊"

        elif prediction == 2:
            sentiment = "Neutral 😐"
        elif prediction == 0:
            sentiment = "Nagative 😞"   

        else:
            sentiment = str(prediction)

        # Result
        st.markdown(
            f"""
            <div class="result-box">
                <div>Predicted Sentiment</div>
                <div class="sentiment">{sentiment}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show processed text
        with st.expander("View processed text"):
            st.write(processed_text)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "CREAT BY BHARGAV"
)
