import numpy as np, pandas as pd, lightgbm as lgb, time
from sklearn.metrics import roc_auc_score
from common import *
tr, te = load(); y = tr[TARGET].values
X = tr[CATS + SERV + NUMS].copy()
for c in CATS: X[c] = X[c].astype('category')
oof = np.zeros(len(y))
for k, (a, b) in enumerate(folds(y)):
    t = time.time()
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, verbose=-1, n_jobs=16)
    m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], eval_metric='auc', callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    print(k, m.best_iteration_, round(roc_auc_score(y[b], oof[b]), 6), f'{time.time()-t:.0f}s', flush=True)
print('OOF AUC', roc_auc_score(y, oof))
imp = pd.Series(m.booster_.feature_importance('gain'), X.columns).sort_values(ascending=False); print(imp.round(0))
