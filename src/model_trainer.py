from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def train_svd(X, n_components=150):
    svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=42)
    movie_vectors = svd.fit_transform(X)
    return svd, movie_vectors

def train_content_model(movies_df):
    movies_df['genres'] = movies_df['genres'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    return tfidf, tfidf_matrix