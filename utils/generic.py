import numpy as np
import pandas as pd
from pathlib import Path

def import_datasets(path=Path('./data/ml-latest-small'), ratings_filename='ratings', movies_filename='movies', filetype='csv'):
    ratings = '.'.join([ratings_filename, filetype])
    movies = '.'.join([ratings_filename, filetype])
    if filetype == 'csv':
        ratings = pd.read_csv(path/ratings)
        movies = pd.read_csv(path/movies)
    elif filetype == 'dat':
        ratings = pd.DataFrame(np.loadtxt(path/ratings, delimiter='::', dtype=int), columns=['userId', 'movieId', 'rating', 'timestamp'])
        movies = pd.DataFrame(np.array([list(row) for row in np.genfromtxt(path/'movies.dat', delimiter='::', encoding='latin1', dtype=None, invalid_raise=False)]), columns=['movieId', 'title', 'genre'])
    else:
        raise ValueError('Only .csv and .dat are supported.')
    return ratings, movies
    