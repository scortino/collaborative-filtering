import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def import_datasets(path=Path('./data/ml-latest-small'), ratings_csv='ratings.csv', movies_csv='movies.csv'):
    ratings = pd.read_csv(path/ratings_csv)
    movies = pd.read_csv(path/movies_csv)
    return ratings, movies

def prepare_data(ratings):
    n_raters = ratings['userId'].nunique()
    n_movies_rated = ratings['movieId'].nunique()
    average_rating_of_movie = ratings.groupby('movieId')['rating'].mean()
    total_raters_of_movie = ratings.groupby('movieId')['userId'].count()
    ratings2 = ratings.merge(average_rating_of_movie, how='inner', on='movieId', sort='userId')
    ratings2 = ratings2.rename(columns={'rating_x':'rating', 'rating_y':'average_rating'})
    ratings2['Normalized_rating'] = ratings2['rating'] - ratings2['average_rating']
    user_movie_matrix = pd.pivot_table(ratings2, values='Normalized_rating', \
                                       index='userId', columns='movieId')
    user_movie_matrix_no_NA = user_movie_matrix.fillna(user_movie_matrix.mean(axis=0))
    cos_similarity = pd.DataFrame(cosine_similarity(user_movie_matrix_no_NA), \
                             index=user_movie_matrix_no_NA.index, \
                             columns=user_movie_matrix_no_NA.index)
    return n_raters, n_movies_rated, user_movie_matrix, user_movie_matrix_no_NA, \
                cos_similarity, average_rating_of_movie

def KNN(df, user, k):
    user_row = df.iloc[user-1]
    neighbors = np.argpartition(user_row, -k-1)[-k-1:].to_numpy() + 1
    return neighbors[neighbors != user]

def check_KNN(df, user1, user2): # User number start from 1
    u1 = df[ratings.userId == user1]
    u2 = df[ratings.userId == user2]
    comparison = u1.merge(u2, how='inner', on='movieId')
    comparison.drop(columns=['userId_x', 'userId_y', 'timestamp_x', 'timestamp_y'], inplace=True)
    comparison = comparison.rename(columns={'rating_x':'rating user {}'.format(user1), \
            'rating_y': 'rating user {}'.format(user2)})
    return comparison

def predict_rating(user_movie_matrix, user, movie, \
                   k, average_rating_of_movie, neighbors=None): # Numbers starting from 1
    if neighbors is None:
        neighbors_of_user = KNN(cos_similarity, user, k)
    else:
        neighbors_of_user = neighbors
    ratings_of_movie = user_movie_matrix_no_NA[movie]
    actual_ratings = ratings_of_movie.iloc[neighbors_of_user-1] + \
                        average_rating_of_movie[movie]
    similarities = cos_similarity.iloc[user-1, neighbors_of_user-1]
    weighted_mean = sum(similarities * actual_ratings) / sum(similarities)
    return weighted_mean

def evaluation(ratings, k, user_movie_matrix_no_NA, average_rating_of_movie, \
               cos_similarity, p=0.2):
    sum_squared_error = 0
    prev_user = 0
    neighbors = []
    i = 0
    for index, row in ratings.iterrows():
        if i > p * ratings.shape[0]:
            break
        user_id, movie_id, rating = int(row['userId']), int(row['movieId']), int(row['rating'])
        if user_id != prev_user:
            neighbors = KNN(cos_similarity, user_id, k)
            prev_user = user_id
        pred = predict_rating(user_movie_matrix_no_NA, user_id, movie_id, k, \
                              average_rating_of_movie, neighbors=neighbors)
        sum_squared_error += (pred - rating)**2
        i += 1
    return np.sqrt(sum_squared_error/(ratings.shape[0]*p))

