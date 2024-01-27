import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import unicodedata
import ftfy
from nltk.tokenize import word_tokenize
import joblib

def tokenize_german_text(text):
    return word_tokenize(text, language='german')

def lowercase_german_text(text):
    text = ftfy.fix_text(text).lower()
    return text

def remove_german_stopwords(text):
    stop_words = set(stopwords.words("german"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(filtered_text)

def handle_special_characters(text):
    normalized_text = unicodedata.normalize("NFKD", text)
    return normalized_text

def preprocess_data(df,num_rows=None):
    save_vectorizer_path=""

    df.dropna(inplace=True)
    
    if num_rows is not None:
        df = df.head(num_rows)

    X = df['text'].values
    y = df['label'].values

    # Apply language-specific pre-processing steps
    X = [lowercase_german_text(text) for text in X]

    X = [remove_german_stopwords(text) for text in X]

    X = [handle_special_characters(text) for text in X]

    X = [' '.join(tokenize_german_text(text)) for text in X]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    tfidf_vectorizer = TfidfVectorizer()

# Vectorize the text data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    if save_vectorizer_path:
        joblib.dump(tfidf_vectorizer, save_vectorizer_path)
        print(f"Vectorizer saved at: {save_vectorizer_path}")

    # Encode labels using LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    return X_train_tfidf, X_test_tfidf, y_train_encoded, y_test_encoded, tfidf_vectorizer, label_encoder

def preprocess_input_text(text):

    text = ftfy.fix_text(text).lower()

    word_tokens = word_tokenize(text, language='german')

    stop_words = set(stopwords.words("german"))

    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]

    text = " ".join(filtered_text)

    text = unicodedata.normalize("NFKD", text)

    return text