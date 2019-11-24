import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class MovieRatingsGenerator(object):
    def __init__(self, ratings_csv_file_path):
        self.ratings = pd.read_csv(ratings_csv_file_path)
        self._prepare_data()
    
    def _prepare_data(self):
        self.n_raters = self.ratings['userId'].nunique()
        self.n_movies_rated = self.ratings['movieId'].nunique()
        self.average_rating_of_movie = self.ratings.groupby('movieId')['rating'].mean()
        ratings_augmented = self.ratings.merge(self.average_rating_of_movie, how='inner', on='movieId', sort='userId')
        ratings_augmented = ratings_augmented.rename(columns={'rating_x':'rating', 'rating_y':'average_rating'})
        ratings_augmented['Normalized_rating'] = ratings_augmented['rating'] - ratings_augmented['average_rating']
        user_movie_matrix = pd.pivot_table(ratings_augmented, values='Normalized_rating', index='userId', columns='movieId')
        self.user_movie_matrix_no_NA = user_movie_matrix.fillna(user_movie_matrix.mean(axis=0))
        self.cos_similarity = pd.DataFrame(cosine_similarity(self.user_movie_matrix_no_NA), \
                                index=self.user_movie_matrix_no_NA.index, \
                                columns=self.user_movie_matrix_no_NA.index)
        return None
    
    def _KNN(self, user, k):
        user_row = self.cos_similarity.iloc[user-1]
        neighbors = np.argpartition(user_row, -k-1)[-k-1:].to_numpy() + 1
        return neighbors[neighbors != user]

    def predict_rating(self, user, movie, k, neighbors=None):
        if neighbors is None:
            neighbors = self._KNN(user, k)
        ratings_of_movie = self.user_movie_matrix_no_NA[movie]
        actual_ratings = ratings_of_movie.iloc[neighbors-1] + self.average_rating_of_movie[movie]
        similarities = self.cos_similarity.iloc[user-1, neighbors-1]
        weighted_mean = sum(similarities * actual_ratings) / sum(similarities)
        return weighted_mean
    
    def evaluate(self, k, frac=0.2):
        sum_squared_error = 0
        for _, row in self.ratings.sample(frac=frac).iterrows():
            user_id, movie_id, rating = int(row['userId']), int(row['movieId']), int(row['rating'])
            pred = self.predict_rating(user_id, movie_id, k)
            sum_squared_error += (pred - rating)**2
        return np.sqrt(sum_squared_error/(self.ratings.shape[0]*frac))
