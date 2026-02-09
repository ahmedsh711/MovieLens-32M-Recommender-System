import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import linear_kernel


class MovieRecommender:
    def __init__(self, model_path):
        self.data = joblib.load(model_path)
        self.svd = self.data['svd']
        self.movie_vectors = self.data['movie_vectors']
        self.user_mapper = self.data['user_mapper']
        self.movie_inv_mapper = self.data['movie_inv_mapper']
        self.movies = self.data['movies_df']
        self.tfidf_matrix = self.data['tfidf_matrix']

    def get_hybrid_recommendations(self, user_id, alpha=0.6, k=10):
        # 1. Cold Start Check
        if user_id not in self.user_mapper:
            return None

        # 2. SVD Calculation
        user_idx = self.user_mapper[user_id]
        user_vec = self.svd.components_[:, user_idx]
        svd_scores = self.movie_vectors.dot(user_vec)

        # 3. Popularity Calculation (Aligned)
        pop_scores = np.zeros(len(svd_scores))
        bayes_lookup = dict(zip(self.movies['movieId'], self.movies['bayesian_avg']))

        for idx, movie_id in self.movie_inv_mapper.items():
            if movie_id in bayes_lookup:
                pop_scores[idx] = bayes_lookup[movie_id]

        # 4. Normalize
        svd_norm = (svd_scores - svd_scores.min()) / (svd_scores.max() - svd_scores.min())
        pop_norm = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min())

        # 5. Blend
        hybrid_scores = (svd_norm * alpha) + (pop_norm * (1 - alpha))

        # 6. Return Top K
        top_indices = np.argsort(hybrid_scores)[::-1][:k]

        results = []
        for idx in top_indices:
            movie_id = self.movie_inv_mapper[idx]
            info = self.movies[self.movies['movieId'] == movie_id].iloc[0]
            results.append({
                'title': info['title'],
                'genres': info['genres'],
                'score': round(hybrid_scores[idx], 3)
            })
        return pd.DataFrame(results)

    def get_similar_movies(self, movie_title, k=10):
        """Content-Based Similarity"""
        matches = self.movies[self.movies['title'].str.contains(movie_title, case=False)]
        if matches.empty:
            return None

        idx = matches.index[0]
        cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        movie_indices = [i[0] for i in sim_scores[1:k + 1]]
        return self.movies.iloc[movie_indices][['title', 'genres', 'bayesian_avg']]