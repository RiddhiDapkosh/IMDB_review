
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Ensure NLTK data is downloaded
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
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

download_nltk_data()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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

def main():
    # Load the dataset
    try:
        df = pd.read_csv("IMDB_Dataset.csv", engine='python')
    except FileNotFoundError:
        print("Error: IMDB_Dataset.csv not found. Please ensure the file is in the same directory.")
        return

    # Drop duplicates
    df = df.drop_duplicates()

    # Split data
    X = df['review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Process text
    X_train_processed = X_train.apply(clean_and_process_text)
    X_test_processed = X_test.apply(clean_and_process_text)

    # Vectorize text
    vectorizer = CountVectorizer(max_features=10000)
    X_train_vectors = vectorizer.fit_transform(X_train_processed)
    X_test_vectors = vectorizer.transform(X_test_processed)

    # Save the vectorizer
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Vectorizer saved as vectorizer.pkl")

    # Train and evaluate the best Logistic Regression model
    print("
--- Training and evaluating Logistic Regression Model ---")
    best_lr_model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
    best_lr_model.fit(X_train_vectors, y_train)
    y_pred_lr = best_lr_model.predict(X_test_vectors)

    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    precision_lr = precision_score(y_test, y_pred_lr, pos_label='positive')
    recall_lr = recall_score(y_test, y_pred_lr, pos_label='positive')
    f1_lr = f1_score(y_test, y_pred_lr, pos_label='positive')

    print(f"Accuracy: {accuracy_lr:.4f}")
    print(f"Precision: {precision_lr:.4f}")
    print(f"Recall: {recall_lr:.4f}")
    print(f"F1-Score: {f1_lr:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_lr))

    # Save the best Logistic Regression model
    joblib.dump(best_lr_model, 'model.pkl')
    print("Best Logistic Regression model saved as model.pkl")

if __name__ == '__main__':
    main()
