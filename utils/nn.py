import numpy as np
import pandas as pd
import torch
from functools import reduce
from sklearn.model_selection import train_test_split
from torch import nn, optim

def log_stepped(start, stop, n):
    mult = stop / start
    step = mult ** (1 / (n - 1))
    return np.array([start * (step ** i) for i in range(n)])

def annealing_cos(start, end, pct):
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end) / 2 * cos_out

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

        self.num_layers = 4 + len(hidden) * 3

        hidden = [2 * n_factors] + hidden
        dropouts = (len(hidden) - len(dense_dropouts) - 1) * [dense_dropouts[0]] + dense_dropouts

        # Network
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.d = nn.Dropout(embedding_dropout)
        self.hidden_fc = nn.Sequential(*reduce(lambda x, y: x + y, [[nn.Linear(i, o), nn.ReLU(), nn.Dropout(d)] for i, o, d in zip(hidden[:-1], hidden[1:], dropouts)]))
        self.last_fc = nn.Linear(hidden[-1], 1)

        self.flattened_modules = list(self.modules())[1:4] + list(self.modules())[5:] # TODO

        self.random_weights()

    def random_weights(self):
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
        self.optimizer = self.init_optim(opt_func)
        self.loss_func = lambda pred, true: torch.sqrt(torch.max(torch.tensor(0.).to(self.data.device), loss_func()(pred, true)))

    def __repr__(self):
        return str(self.model)

    def lr_range(self, lr):
        if isinstance(lr, float) or isinstance(lr, int):
            return [lr] * self.model.num_layers
        if isinstance(lr, (list, tuple)):
            return (self.model.num_layers - len(lr)) * [lr[0]] + list(lr)
        if not isinstance(lr, slice):
            return r
        if lr.start:
            res = log_stepped(lr.start, lr.stop, self.model.num_layers - 1)
            return [res[0]] + res
        else:
            return np.array([lr.stop / 10] * (self.model.num_layers - 1) + [lr.stop])

    def init_optim(self, opt_func):
        optimizer = opt_func([{'params': module.parameters()} for module in self.model.flattened_modules])
        assert len(optimizer.param_groups) == self.model.num_layers
        return optimizer

    def set_param_per_layer(self, param_name, params):
        self.optimizer.param_groups = [{**self.optimizer.param_groups[i], **{param_name: params[i]}} for i in range(self.model.num_layers)]
    
    def fit(self, epochs, lr=slice(1e-3), wd=1e-5, scheduler=None):
        self.epochs = epochs
        self.lr = self.lr_range(lr)
        self.set_param_per_layer('lr', self.lr)
        self.set_param_per_layer('weight_decay', [wd] * self.model.num_layers)
        if scheduler is None:
            scheduler = type('dummy_scheduler', (object,), {'initialize': lambda: None, 'update': lambda: None})
        scheduler.initialize()
        for epoch in range(epochs):
            running_loss = dict()
            for stage in ('train', 'val'):
                train = (stage == 'train')
                running_loss[stage] = 0
                for X_batch, y_batch in self.data.make_batches(stage=stage, shuffle=train):
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(train):
                        # forward
                        out = self.model(X_batch[:, 0], X_batch[:, 1], y_range=self.data.y_range)
                        loss = self.loss_func(out, y_batch.float())

                        if stage == 'train':
                            # backward
                            loss.backward()

                            # update parameters
                            self.optimizer.step()

                            # update lr and moms
                            scheduler.update()

                    running_loss[stage] += loss

            print(f"epoch: {epoch}, train loss: {running_loss['train'] / self.data.n_batches['train']}, validation loss: {running_loss['val'] / self.data.n_batches['val']}")

    def fit_one_cycle(self, cycle_len, lr_max=slice(1e-3), moms=[0.95, 0.85], wd=1e-5, **kwargs):
        lr_max = self.lr_range(lr_max)
        scheduler = OneCycleScheduler(self, lr_max=lr_max, moms=moms, **kwargs)
        self.fit(epochs=cycle_len, lr=lr_max, wd=wd, scheduler=scheduler)
            
    def get_preds(self, ds_type='val'):
        return np.concatenate((self.model(self.data.data[ds_type]['X'][:, 0], self.data.data[ds_type]['X'][:, 1]).detach().cpu().numpy()[:, np.newaxis], self.data.data[ds_type]['y'].cpu().numpy()[:, np.newaxis]), axis=1)

class Scheduler():
    def __init__(self, start, end=0, n_iter=1, annealing_func=annealing_cos):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.func = annealing_func
        self.n = 0

    def step(self):
        "Next value along annealed schedule."
        self.n += 1
        return self.func(self.start, self.end, self.n / self.n_iter)

class OneCycleScheduler:
    def __init__(self, learner, lr_max, moms=(0.95, 0.85), div_factor=25., pct_start=0.3, final_div=None):
        self.learn = learner
        self.lr_max = np.array(lr_max) # check if float or list
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.final_div = final_div if final_div is not None else div_factor * 1e4
        self.moms = list(moms)
    
    def steps(self, *phases_config):
        "Build anneal schedules."
        return [Scheduler(start, end, n_iter, annealing_func) for start, end, n_iter, annealing_func in phases_config]

    def initialize(self):
        "Initialize optimizer's parameters based on annealing schedule."
        n = self.learn.data.n_batches['train'] * self.learn.epochs # total number of batches
        a1 = int(n * self.pct_start) # number of batches with a lr increase
        a2 = n - a1 # number of batches with a lr decrease
        low_lr = self.lr_max / self.div_factor # phase 1 from low_lr to lr_max
        lowest_lr = self.lr_max / self.final_div # phase 2 from lr_max to lowest_lr
        self.lr_schedules = self.steps((low_lr, self.lr_max, a1, annealing_cos), (self.lr_max, lowest_lr, a2, annealing_cos)) # change annealing func here
        self.mom_schedules = self.steps((*self.moms, a1, annealing_cos), (*self.moms[::-1], a2, annealing_cos))
        self.optimizer = self.learn.optimizer
        self.learn.set_param_per_layer('lr', self.lr_schedules[0].start)
        self.learn.set_param_per_layer('momentum', [self.mom_schedules[0].start] * self.learn.model.num_layers)
        self.phase = 0 # keep count of phase we are on, one schedule per phase

    def update(self):
        "Change optimizer's parameters according to annealing schedule."
        self.learn.set_param_per_layer('lr', self.lr_schedules[self.phase].step())
        self.learn.set_param_per_layer('momentum', [self.mom_schedules[self.phase].step()] * self.learn.model.num_layers)
        if self.lr_schedules[self.phase].n >= self.lr_schedules[self.phase].n_iter:
            self.phase += 1 # move onto next phase
