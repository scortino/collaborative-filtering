import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class MovieRatingsGenerator5(object):
    def __init__(self, data):
        self.ratings = data
        self._prepare_data()
    
    def _prepare_data(self):
        self.n_raters = self.ratings['userId'].nunique()
        self.n_movies_rated = self.ratings['movieId'].nunique()
        self.average_rating_of_movie = self.ratings.groupby('movieId')['rating'].mean()
        self.average_rating_of_user = self.ratings.groupby('userId')['rating'].mean()
        ratings_augmented = self.ratings.merge(self.average_rating_of_movie, how='inner', on='movieId', sort='userId')
        ratings_augmented = ratings_augmented.rename(columns={'rating_x':'rating', 'rating_y':'average_rating'})
        ratings_augmented['Normalized_rating'] = ratings_augmented['rating']
        for user in self.average_rating_of_user.index:
            ratings_augmented.loc[ratings_augmented.userId == user, 'Normalized_rating'] -= self.average_rating_of_user[user]
        user_movie_matrix = pd.pivot_table(ratings_augmented, values='Normalized_rating', index='userId', columns='movieId')
        self.user_movie_matrix_no_NA = user_movie_matrix.fillna(user_movie_matrix.mean(axis=0))
        self.cos_similarity = pd.DataFrame(cosine_similarity(self.user_movie_matrix_no_NA), \
                                index=self.user_movie_matrix_no_NA.index, \
                                columns=self.user_movie_matrix_no_NA.index)
        return None
    
 
    def _KNN(self, user, k):
        if user in self.cos_similarity.index:
            if k >= self.cos_similarity.shape[0]:
                return self.cos_similarity.index[self.cos_similarity.index != user]
            user_row = self.cos_similarity[user]
            sorted_neighbors = user_row.sort_values(ascending=False).iloc[1:k+1]
            return sorted_neighbors.index
        else:
            return []


    def predict_rating(self, user, movie, k, neighbors=None):
        if neighbors is None:
            neighbors = self._KNN(user, k)
        if neighbors == []:
            if movie in self.average_rating_of_movie.index:
                # print("Case 2")
                return self.average_rating_of_movie[movie]
            else:
                # print("Case 4")
                return 3

        if movie in self.user_movie_matrix_no_NA.columns:
            ratings_of_movie = self.user_movie_matrix_no_NA[movie]
            actual_ratings = ratings_of_movie[neighbors]
        else:
            # print("Case 3")
            return self.average_rating_of_user[user]
        
        similarities = self.cos_similarity[user][neighbors]
        if sum(similarities) == 0:
            return sum(actual_ratings)/k + self.average_rating_of_user[user]
        weighted_mean = sum(similarities * actual_ratings) / sum(similarities) + self.average_rating_of_user[user]
        # print("Case 1")
        return weighted_mean
    
    def evaluate(self, k, testing_data):
        sum_squared_error = 0
        for _, row in testing_data.iterrows():
            user_id, movie_id, rating = int(row['userId']), int(row['movieId']), int(row['rating'])
            pred = self.predict_rating(user_id, movie_id, k)
            sum_squared_error += (pred - rating)**2
        return np.sqrt(sum_squared_error/(testing_data.shape[0]))



# We handle 4 cases:
# Case 1 - both user and movie are known -> Weighted mean
# Case 2 - User unknown, movie known -> Average rating of movie
# Case 3 - User known, movie unknown -> Average rating of user
# Case 4 - Both unknown -> 3
    
# Additionally, we check for the case when sum(similarities) == 0,
# which is a subcase of case 1.