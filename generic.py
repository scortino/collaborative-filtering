import pandas as pd
from pathlib import Path

def import_datasets(path=Path('./data/ml-latest-small'), ratings_csv='ratings.csv', movies_csv='movies.csv'):
    ratings = pd.read_csv(path/ratings_csv)
    movies = pd.read_csv(path/movies_csv)
    return ratings, movies
    