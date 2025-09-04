# 📚 Amazon Book Review Rating Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-green)
![Docker](https://img.shields.io/badge/Docker-20.10-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

This project predicts **Amazon book review star ratings (1–5)** directly from review text.  
Unlike traditional sentiment classification, this is framed as a **regression task**, estimating a continuous score that aligns with the reviewer’s rating.

The work began as Jupyter notebooks for experimentation and was later converted into production-ready **Python scripts + FastAPI API** for deployment.

---

## 🚀 Features
- End-to-end **NLP pipeline**: text cleaning, tokenization, lemmatization.
- Multiple **word embedding methods**: TF-IDF, Word2Vec, GloVe, fastText.
- Multiple **models**: LightGBM, MLP, RNN, LSTM.
- RESTful **FastAPI service** for real-time predictions.
- **Docker support** for easy deployment.

---

## 📊 Dataset
- Source: **Amazon Customer Reviews (Books category)**    
- ~160,000 samples used after filtering only **verified purchases**.  
- Key fields used:
  - `review_headline`  
  - `review_body`  
  - `star_rating` (target)  

---

## 🛠️ Preprocessing
- Lowercasing & punctuation removal.  
- HTML tag cleanup.  
- Contraction expansion (e.g., *don’t → do not*).  
- Stopword removal (while preserving negations).  
- Lemmatization (via spaCy).  

---

## 🧠 Models & Results
| Embedding + Model     | RMSE   |
|------------------------|--------|
| TF-IDF + LightGBM      | 0.9497 |
| Word2Vec + LightGBM    | 0.9239 |
| GloVe + LightGBM       | 1.0369 |
| GloVe + MLP            | 0.9733 |
| **fastText + MLP**     | **0.8501 (best)** |
| GloVe + LSTM           | 0.8604 |
| GloVe + SimpleRNN      | 1.2270 |

✅ The best configuration was **fastText embeddings + MLP**, achieving RMSE = **0.8501**, showing the advantage of subword-level embeddings.

---

## 📂 Project Structure
```
BOOK_PROJECT/
│── app.py # FastAPI app for serving predictions
│── main.py # Training pipeline (train embeddings + model)
│── model.py # Helper functions (data processing, model training)
│── requirements.txt # Python dependencies
│── Dockerfile # Containerization setup
│── report/Book Rating Report.pdf # Full project report
│── data/ # (ignored) raw Amazon review data
```

---

## ⚡ Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models
```bash
python main.py --data_path data/amazon_reviews_us_Books_v1_02.tsv \
               --mlp_output mlp_model.joblib \
               --ft_output fasttext.model \
               --epochs 50 --batch_size 256
```

### 3. Run FastAPI server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API available at:  
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Example request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"review": "This book was amazing, I loved every part of it!"}'
```

### 🐳 Docker Deployment
Build the image:
```bash
docker build -t book-rating-app .
```

Run the container:
```bash
docker run -it --rm -p 8000:8000 book-rating-app
```

---

## 🔮 Future Work
- Experiment with Transformer-based models (BERT, DistilBERT) for contextual embeddings.
- Deploy the API to cloud services (AWS/GCP/Heroku).
- Build a simple web frontend to demo predictions interactively.

---

## 📘 References
- Stanford CS224n: Word2Vec & GloVe Notes
- Viktoria Vida, *Product review prediction of Amazon products using ML models*

---

## ✨ Author
Dinh Dang Khoa Tran  
University of Illinois at Chicago (UIC), Computer Science (Class of 2027)