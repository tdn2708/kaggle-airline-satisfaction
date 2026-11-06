"""Export the feature matrix (base + route stats + teacher) for Colab GPU training."""
import numpy as np, pandas as pd
from common import *; from feats import *
tr, te = load(); y = tr[TARGET].values; o = load_orig()
R_tr, R_te = route_stats(tr, te, o); ta, tb = teacher(tr, te, o, R_tr, R_te)
def mk(df, R, t):
    X = pd.concat([df[CATS + SERV + NUMS], R], axis=1); X['teacher'] = t
    for c in CATS: X[c] = X[c].astype(str)
    return X
X, Xt = mk(tr, R_tr, ta), mk(te, R_te, tb)
fold = np.zeros(len(y), int)
for k, (_, b) in enumerate(folds(y)): fold[b] = k
X['__y'] = y; X['__fold'] = fold; Xt['__id'] = te['id'].values
X.to_parquet(ROOT / 'colab/train_feats.parquet'); Xt.to_parquet(ROOT / 'colab/test_feats.parquet')
print(X.shape, Xt.shape)
