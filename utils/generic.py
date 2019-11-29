import numpy as np
import os
import pandas as pd
from pathlib import Path

def import_datasets(path=Path('./data/ml-latest-small'), ratings='ratings.csv', movies='movies.csv'):
    ratings_ext = os.path.splitext(ratings)[1] 
    if ratings_ext == '.csv':
        ratings = pd.read_csv(path/ratings)
        movies = pd.read_csv(path/movies)
    elif ratings_ext == '.dat':
        ratings = pd.DataFrame(np.loadtxt(path/ratings, delimiter='::', dtype=int), columns=['userId', 'movieId', 'rating', 'timestamp'])
        movies = pd.DataFrame(np.array([list(row) for row in np.genfromtxt(path/'movies.dat', delimiter='::', encoding='latin1', dtype=None, invalid_raise=False)]), columns=['movieId', 'title', 'genre'])
    else:
        ratings = pd.read_csv(path/ratings, delimiter='\t', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
        movies = pd.read_csv(path/movies, delimiter='|', usecols=[0, 1], encoding='latin-1', header=None, names=['movieId', 'title'])
    return ratings, movies
    