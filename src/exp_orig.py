import sys, numpy as np, pandas as pd, lightgbm as lgb, time
from sklearn.metrics import roc_auc_score
from common import *; from feats import *
variant = sys.argv[1]
tr, te = load(); y = tr[TARGET].values; o = load_orig()
X = base(tr)
if 'te' in variant: X = pd.concat([X, orig_te(tr, o)], axis=1)
if 'ex' in variant: X = pd.concat([X, extra(tr)], axis=1)
Xo = None
if 'cat' in variant:  # append original rows to each training fold
    Xo = base(o)
    if 'te' in variant: Xo = pd.concat([Xo, orig_te(o, o)], axis=1)
    if 'ex' in variant: Xo = pd.concat([Xo, extra(o)], axis=1)
    for c in CATS: Xo[c] = pd.Categorical(Xo[c], categories=X[c].cat.categories)
    Xo['is_orig'] = 1; X['is_orig'] = 0
oof = np.zeros(len(y))
for k, (a, b) in enumerate(folds(y)):
    if k >= int(sys.argv[2] if len(sys.argv) > 2 else 5): break
    Xa, ya = X.iloc[a], y[a]
    if Xo is not None: Xa = pd.concat([Xa, Xo]); ya = np.r_[ya, o[TARGET].values]
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, verbose=-1, n_jobs=16)
    m.fit(Xa, ya, eval_X=(X.iloc[b],), eval_y=(y[b],), eval_metric='auc', callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    print(variant, k, m.best_iteration_, round(roc_auc_score(y[b], oof[b]), 6), flush=True)
