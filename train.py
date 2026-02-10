import joblib
from pathlib import Path
from src.data_processor import load_and_process_data
from src.model_trainer import train_svd, train_content_model

# Config
DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Data Processing
    X, movies, user_mapper, movie_inv, user_means = load_and_process_data(DATA_DIR)

    # 2. Train SVD (Collaborative)
    svd, movie_vectors = train_svd(X, n_components=150)

    # 3. Train TF-IDF (Content)
    tfidf, tfidf_matrix = train_content_model(movies)

    # 4. Save Artifacts
    artifacts = {
        'svd': svd,
        'movie_vectors': movie_vectors,
        'user_mapper': user_mapper,
        'movie_inv_mapper': movie_inv,
        'user_means': user_means,
        'movies_df': movies,
        'tfidf_matrix': tfidf_matrix,
        'tfidf_vectorizer': tfidf
    }

    save_path = MODEL_DIR / "recommender_artifacts.pkl"
    joblib.dump(artifacts, save_path, compress=3)
    print("Training Complete!")


if __name__ == "__main__":
    main()