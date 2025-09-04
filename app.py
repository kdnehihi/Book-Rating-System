from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from gensim.models import FastText
from model import text_preprocessing, lemmatization, word_tokenization

app = FastAPI()

# Load models
MLP_MODEL_PATH = "mlp_model.joblib"
FT_MODEL_PATH = "fasttext.model"

mlp_model = joblib.load(MLP_MODEL_PATH)
fasttext_model = FastText.load(FT_MODEL_PATH)

class ReviewRequest(BaseModel):
    review: str

def vectorize_review(text: str, fasttext_model):
    cleaned = text_preprocessing(text)
    lemmatized = lemmatization(cleaned)
    tokens = word_tokenization(lemmatized)
    vectors = [fasttext_model.wv[w] for w in tokens if w in fasttext_model.wv]
    if vectors:
        return np.mean(vectors, axis=0).reshape(1, -1)
    else:
        return np.zeros((1, fasttext_model.vector_size))

@app.post("/predict")
async def predict_rating(review: ReviewRequest):
    X = vectorize_review(review.review, fasttext_model)
    prediction = mlp_model.predict(X)[0]
    prediction = float(np.clip(prediction, 0.0, 5.0))
    return {"predicted_star": round(prediction, 2)}
