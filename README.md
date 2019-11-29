# Collaborative Filtering Project

Comparison of KNN and NN based approaches to movie recommendation.

## Installation

Pip:
```
pip install -r requirements.txt
```

## Directory Content

The project directory is composed of the following files:
- `data`:
	- `images`: losses graphs from NN training;
	- `ml-100k`, `ml-1m`, `ml-latest-small`: open source MovieLens datasets from [GroupLens](https://grouplens.org/datasets/movielens/).
- `utils`:
	- `generic.py`: function to import dataset as pandas.DataFrame's into a Jupyter notebook;
	- `knn.py`: classes and functions implementing a KNN-based collaborative filtering model;
	- `nn.py`: classes and functions implementing a NN-based collaborative filtering model.
- `collaborative_filtering.ipynb`: Jupyter notebook reporting preliminary results of the models on the MovieLens datasets.	
