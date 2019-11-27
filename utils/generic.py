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

class CollabData:
    def __init__(self, df, cols=['userId', 'movieId', 'rating'], test_size=0.2):
        unique_users = np.unique(df[cols[0]])
        u_to_i = {u: i for i, u in enumerate(unique_users)}
        self.n_users = len(unique_users)

        unique_movies = np.unique(df[cols[1]])
        m_to_i = {u: i for i, u in enumerate(unique_movies)}
        self.n_movies = len(unique_movies)

        self.X = list(map(lambda x: [u_to_i[x[0]], m_to_i[x[1]]], zip(df[cols[0]], df[cols[1]])))
        self.y = df[cols[2]].values
        
    def show_batch(self, n=10, random_state=None):
        np.random.seed(random_state)
        inds = np.random.permutation(self.X.shape[0])[:n]
        print(pd.DataFrame(np.concatenate((self.X[inds, :], self.y[inds][:, np.newaxis]), axis=1), columns=['userId', 'movieId', 'rating']))

class KNNCollabData(CollabData):
    def __init__(self, df, cols=['userId', 'movieId', 'rating'], test_size=0.2):
        super().__init__(self, df, cols, test_size)

        # TODO

class NNCollabData(CollabData):
    def __init__(self, df, cols=['userId', 'movieId', 'rating'], test_size=0.2, bs=256, random_state=None):
        super().__init__(df, cols, test_size)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        data = dict([('train', dict()), ('val', dict())])
        data['train']['X'], data['val']['X'], data['train']['y'], data['val']['y'] = list(map(lambda x: torch.tensor(x).to(self.device), train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)))
        self.data = data
        
        self.sizes = dict([('train', data['train']['X'].shape[0]), ('val', data['val']['X'].shape[0])])
        self.y_range = [np.min(y), np.max(y)]
        
        self.bs = bs
        self.n_batches = dict([('train', np.ceil(self.sizes['train'] /bs)), ('val', np.ceil(self.sizes['val'] / bs))])
        
    def show_batch(self, n=10, random_state=None):
        np.random.seed(random_state)
        inds = np.random.permutation(self.sizes['train'])[:n]
        data = self.data['train']
        print(pd.DataFrame(np.concatenate((data['X'].cpu()[inds, :], data['y'].cpu()[inds][:, np.newaxis]), axis=1), columns=['userId', 'movieId', 'rating']))
        
    def make_batches(self, stage='train', shuffle=True):
        bs = self.bs
        shuffled_inds = np.random.permutation(self.data[stage]['X'].shape[0]) if shuffle else np.arange(self.data[stage]['X'].shape[0])
        for i in range(int(self.data[stage]['X'].shape[0] / bs) + 1):
            inds = shuffled_inds[bs*i:bs*(i+1)]
            yield self.data[stage]['X'][inds, :].to(self.device), self.data[stage]['y'][inds].to(self.device)
    