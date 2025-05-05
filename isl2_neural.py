import numpy as np, pandas as pd
from matplotlib.pyplot import subplots
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from sklearn.model_selection import train_test_split, GridSearchCV

import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset

from ISLP.torch import SimpleDataModule, SimpleModule, ErrorTracker, rec_num_workers
from ISLP.torch.imdb import load_lookup, load_tensor, load_sparse, load_sequential
from bow import get_vocab, popular_words, preprocess
import spicy
from scipy.sparse import coo_matrix


def one_hot(sequences, ncol):
    idx, vals = [], []
    for i, s in enumerate(sequences):
        idx.extend({(i, v): 1 for v in s}.keys())
    idx = np.array(idx).T
    vals = np.ones(idx.shape[1], dtype=np.float32)
    tens = torch.sparse_coo_tensor(
        indices=idx, values=vals, size=(len(sequences), ncol)
    )
    return tens.coalesce()


def convert_sparse_tensor(X):
    idx = np.asarray(X.indices())
    vals = np.asarray(X.values())
    return coo_matrix((vals, (idx[0], idx[1])), shape=X.shape).tocsr()


df_hw = pd.read_csv("data/best_picture_metadata_with_reviews_filtered.csv")


df_hw["clean"] = df_hw["Description"].apply(lambda x: preprocess(x))
text = " ".join(df_hw["Description"].astype(str))
common_words = get_vocab(preprocess(text), 100)

index_map = {item: idx for idx, item in enumerate(common_words, start=1)}

df_hw["encoded"] = df_hw["clean"].apply(
    lambda lst: [index_map.get(item, 0) for item in lst]
)


train = df_hw[df_hw["Year Nominated"] != 2021]

train = train.copy()
test = df_hw[df_hw["Year Nominated"] == 2021]
test = test.copy()


S_train = list(train["encoded"])


S_test = list(test["encoded"])
X_train = one_hot(S_train, 100)
X_test = one_hot(S_test, 100)
L_train = train["Won"].astype(np.float32)
L_test = test["Won"].astype(np.float32)


print(X_train.shape)


"""
X_train_s = convert_sparse_tensor(X_train)
X_test_s = convert_sparse_tensor(X_test)





X_train_d = torch.tensor(X_train_s.todense())
X_test_d = torch.tensor(X_test_s.todense())


torch.save(X_train_d, 'lbxd_X_train.tensor')
torch.save(X_test_d, 'lbxd_X_test.tensor')


save_npz('lbxd_X_test.npz', X_test_s)
save_npz('lbxd_X_train.npz', X_train_s)

np.save('IMDB_Y_test.npy', L_test)
np.save('IMDB_Y_train.npy', L_train)"""
