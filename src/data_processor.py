import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path


def load_and_process_data(data_dir: Path):
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    movies = pd.read_parquet(data_dir / "movies.parquet")

    # 1. Optimize Types
    ratings['userId'] = ratings['userId'].astype('int32')
    ratings['movieId'] = ratings['movieId'].astype('int32')
    ratings['rating'] = ratings['rating'].astype('float32')

    # 2. Calculate Bayesian Average (Popularity)
    movie_stats = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
    C = movie_stats['count'].mean()
    m = movie_stats['mean'].mean()

    movie_stats['bayesian_avg'] = (
                                          (C * m) + (movie_stats['mean'] * movie_stats['count'])
                                  ) / (C + movie_stats['count'])

    movies = movies.merge(movie_stats, on='movieId', how='left')
    movies['bayesian_avg'] = movies['bayesian_avg'].fillna(m)

    # 3. Center Ratings (Crucial for SVD accuracy!)
    user_means = ratings.groupby('userId')['rating'].mean()
    ratings['centered_rating'] = ratings['rating'] - ratings['userId'].map(user_means)

    # 4. Create Sparse Matrix
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()

    user_mapper = {id: i for i, id in enumerate(user_ids)}
    movie_inv_mapper = {i: id for i, id in enumerate(movie_ids)}
    movie_mapper = {id: i for i, id in enumerate(movie_ids)}

    row_ind = [movie_mapper[i] for i in ratings['movieId']]
    col_ind = [user_mapper[i] for i in ratings['userId']]
    data = ratings['centered_rating'].values  # Use Centered!

    X = csr_matrix((data, (row_ind, col_ind)), shape=(len(movie_ids), len(user_ids)), dtype=np.float32)

    return X, movies, user_mapper, movie_inv_mapper, user_means