import argparse
import pandas as pd
import joblib

from model import (
    read_data, extract_features,
    train_fasttext_model, get_fasttext_vector, train_mlp_model
)

def main(args):
    print("📥 Reading data...")
    df = read_data(args.data_path)
    df = extract_features(df)
    y = df['star_rating'].values.astype(float)
    print("🤖 Training FastText embedding model...")
    full_text = df['review_headline'] + " " + df['review_body']
    fasttext_model = train_fasttext_model(full_text)

    print("🔍 Generating vectors from FastText...")
    X = full_text.apply(lambda x: get_fasttext_vector(x, fasttext_model))
    
    print("✂️ Splitting train/test sets...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    print("🚀 Training MLP regression model...")
    mlp = train_mlp_model(X_train.values, y_train, input_dim=100, epochs=args.epochs, batch_size=args.batch_size)

    print("💾 Saving models...")
    joblib.dump(mlp, args.mlp_output)
    fasttext_model.save(args.ft_output)
    print(f"✅ Saved: {args.mlp_output}, {args.ft_output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="/Users/khoatran/coding/NLP/book-rating-predictor/data/amazon_reviews_us_Books_v1_02.tsv")
    p.add_argument("--mlp_output", default="mlp_model.joblib")
    p.add_argument("--ft_output", default="fasttext.model")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)

    args = p.parse_args()
    main(args)
