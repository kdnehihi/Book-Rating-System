import os
import re
import pandas as pd
import string
from bs4 import BeautifulSoup
from contractions_data import contractions_dict
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import spacy
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

contractions_re = re.compile('(%s)' % '|'.join(re.escape(key) for key in contractions_dict.keys()))
stop_words = set(stopwords.words('english'))
stop_words -= {'not', 'no'}

def expand_contractions(text):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def read_data(file_path):
    return pd.read_csv(file_path, sep="\t", on_bad_lines="skip")

def extract_features(df):
    df = df.sample(n=50000, random_state=42)
    df = df[['star_rating', 'review_headline', 'review_body']]
    df.dropna(inplace=True)
    return df

def text_preprocessing(text):
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = expand_contractions(text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def lemmatization(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])

def word_tokenization(text):
    return text.split()

def train_fasttext_model(text_series):
    processed_tokens = []
    for text in text_series:
        cleaned = text_preprocessing(text)
        lemmatized = lemmatization(cleaned)
        tokens = word_tokenization(lemmatized)
        processed_tokens.append(tokens)

    model = FastText(
        sentences=processed_tokens,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
        seed=42
    )
    return model

def get_fasttext_vector(text, fasttext_model):
    cleaned = text_preprocessing(text)
    lemmatized = lemmatization(cleaned)
    tokens = word_tokenization(lemmatized)
    vectors = [fasttext_model.wv[w] for w in tokens if w in fasttext_model.wv]
    if vectors:
        return pd.Series(sum(vectors) / len(vectors))
    else:
        return pd.Series([0.0] * fasttext_model.vector_size)

def train_mlp_model(X_train, y_train, input_dim=100, epochs=50, batch_size=256):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer=Adam(), loss='mse')

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    return model
