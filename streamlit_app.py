
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK data is downloaded
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt_tab') # Add download for punkt_tab
    except LookupError:
        nltk.download('punkt_tab')

download_nltk_data()

# Initialize lemmatizer and stopwords (outside the function for caching)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning and preprocessing function
def clean_and_process_text(text):
    # 1. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # 3. Remove special characters and numbers (keeping only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 4. Convert to lowercase
    text = text.lower()
    # 5. Tokenize the text
    tokens = word_tokenize(text)
    # 6. Remove stopwords and lemmatize
    filtered_and_lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # 7. Join tokens back into a string and normalize extra spaces
    return " ".join(filtered_and_lemmatized_tokens).strip()

# Load the vectorizer and the best model
@st.cache_resource
def load_resources():
    try:
        vectorizer = joblib.load('vectorizer.pkl')
        model = joblib.load('model.pkl') # This is the best Logistic Regression model
        return vectorizer, model
    except FileNotFoundError:
        st.error("Error: Model or vectorizer files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
        st.stop()

vectorizer, model = load_resources()

# Streamlit App
st.title("IMDB Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to predict its sentiment (positive/negative).")

# Text input
user_input = st.text_area("", height=200, placeholder="Type your movie review here...")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        processed_text = clean_and_process_text(user_input)

        # Vectorize the processed text
        vectorized_text = vectorizer.transform([processed_text])

        # Make prediction
        prediction = model.predict(vectorized_text)
        prediction_proba = model.predict_proba(vectorized_text)

        st.subheader("Prediction:")
        if prediction[0] == 'positive':
            st.success(f"The review is **Positive** with a probability of {prediction_proba[0][1]:.2f}")
        else:
            st.error(f"The review is **Negative** with a probability of {prediction_proba[0][0]:.2f}")

        st.subheader("Details:")
        st.write(f"Processed Text: {processed_text}")

    else:
        st.warning("Please enter a movie review to analyze.")

