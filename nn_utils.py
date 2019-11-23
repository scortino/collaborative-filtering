import numpy as np
import pandas as pd
import torch
from functools import reduce
from sklearn.model_selection import train_test_split
from torch import nn, optim

class CollabData:
    def __init__(self, df, cols=['userId', 'movieId', 'rating'], test_size=0.2, bs=256, random_state=None):
        unique_users = np.unique(df[cols[0]])
        u_to_i = {u: i for i, u in enumerate(unique_users)}
        self.n_users = len(unique_users)

        unique_movies = np.unique(df[cols[1]])
        m_to_i = {u: i for i, u in enumerate(unique_movies)}
        self.n_movies = len(unique_movies)

        X = list(map(lambda x: [u_to_i[x[0]], m_to_i[x[1]]], zip(df[cols[0]], df[cols[1]])))
        y = df[cols[2]].values
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        data = dict([('train', dict()), ('val', dict())])
        data['train']['X'], data['val']['X'], data['train']['y'], data['val']['y'] = list(map(lambda x: torch.tensor(x).to(self.device), train_test_split(X, y, test_size=test_size, random_state=random_state)))
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


class EmbedNet(nn.Module):
    def __init__(self, n_users: int, n_movies: int,
                 n_factors: int=50, embedding_dropout:float =0.02,
                 hidden:list =[10], dense_dropouts:list =[0.2]):
                
        super().__init__()
        
        hidden = [2 * n_factors] + hidden
        dropouts = (len(hidden) - len(dense_dropouts) - 1) * [dense_dropouts[0]] + dense_dropouts
        
        # Network
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.d = nn.Dropout(embedding_dropout)
        self.hidden_fc = nn.Sequential(*reduce(lambda x, y: x + y, [[nn.Linear(i, o), nn.ReLU(), nn.Dropout(d)] for i, o, d in zip(hidden[:-1], hidden[1:], dropouts)]))
        self.last_fc = nn.Linear(hidden[-1], 1)
        
        # Initialize embeddings
        self.u.weight.data.normal_()
        self.m.weight.data.normal_()
        
        # Initialize weights hidden fully connected layers
        for linear_layer in self.hidden_fc[::3]:
            nn.init.xavier_uniform_(linear_layer.weight)
            linear_layer.bias.data.fill_(0.01)
        
        # Initialize weights last layer
        nn.init.xavier_uniform_(self.last_fc.weight)
        self.last_fc.bias.data.fill_(0.01)
    
    def forward(self, users, movies, y_range: list=[0, 5]):
        x = torch.cat([self.u(users), self.m(movies)], axis=1)
        x = self.d(x)
        x = self.hidden_fc(x)
        x = self.last_fc(x)
        out = torch.squeeze(torch.sigmoid(x) * (y_range[1] - y_range[0] + 1) + y_range[0] - 0.5)
        return out

class CollabLearner:
    def __init__(self, data: CollabData, arch=EmbedNet, n_factors=50, opt_func=optim.AdamW, loss_func=nn.MSELoss, **kargs):
        self.data = data
        self.model = arch(data.n_users, data.n_movies, n_factors, **kargs).to(self.data.device)
        self.opt_func = opt_func
        self.loss_func = lambda pred, true: torch.sqrt(loss_func()(pred, true))
        
    def __repr__(self):
        return str(self.model)
    
    def fit(self, epochs, lr=1e-3, wd=1e-5):
        optimizer = self.opt_func(params=self.model.parameters(), lr=lr, weight_decay=wd)
        for epoch in range(epochs):
            running_loss = dict()
            for stage in ('train', 'val'):
                flag = (stage == 'train')
                running_loss[stage] = 0
                for X_batch, y_batch in self.data.make_batches(stage=stage, shuffle=flag):
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(flag):
                        # forward
                        out = self.model(X_batch[:, 0], X_batch[:, 1], y_range=self.data.y_range)
                        loss = self.loss_func(out, y_batch.float())

                        if stage == 'train':
                            # backward
                            loss.backward()

                            # update parameters
                            optimizer.step()

                    running_loss[stage] += loss

            print(f"epoch: {epoch}, train loss: {running_loss['train'] / self.data.n_batches['train']}, validation loss: {running_loss['val'] / self.data.n_batches['val']}")
            
    def get_preds(self, ds_type='val'):
        return np.concatenate((self.model(self.data.data[ds_type]['X'][:, 0], self.data.data[ds_type]['X'][:, 1]).detach().cpu().numpy()[:, np.newaxis], self.data.data[ds_type]['y'].cpu().numpy()[:, np.newaxis]), axis=1)
