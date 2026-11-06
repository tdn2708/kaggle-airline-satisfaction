"""Level-2 LightGBM stacker on model logits (+ optional segment columns), nested 5-fold CV vs logistic stacking."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from common import *
ZOO = ART / 'zoo'
names = [a for a in sys.argv[1:] if not a.startswith('--')] or sorted(p.name[:-8] for p in ZOO.glob('*_oof.npy'))
ctx = '--ctx' in sys.argv
tr, te = load(); y = tr[TARGET].values
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
O = pd.DataFrame({n: L(np.load(ZOO / f'{n}_oof.npy')) for n in names})
T = pd.DataFrame({n: L(np.load(ZOO / f'{n}_test.npy')) for n in names})
if ctx:
    for c in ['Class', 'Type of Travel', 'Customer Type']:
        O[c] = tr[c].astype('category').cat.codes; T[c] = te[c].astype('category').cat.codes
    O['route_cnt'] = tr.groupby('Flight Distance')['id'].transform('size').values
    T['route_cnt'] = te['Flight Distance'].map(tr.groupby('Flight Distance').size()).fillna(0).values
P = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=500, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=10, verbose=-1, n_jobs=6)
res = {'logreg': np.zeros(len(y)), 'lgb': np.zeros(len(y))}
for a, b in StratifiedKFold(5, shuffle=True, random_state=123).split(O, y):
    res['logreg'][b] = LogisticRegression(max_iter=2000).fit(O[names].iloc[a], y[a]).decision_function(O[names].iloc[b])
    res['lgb'][b] = lgb.LGBMClassifier(**P).fit(O.iloc[a], y[a]).predict_proba(O.iloc[b])[:, 1]
for k, v in res.items(): print(f'{k:8s} nested {roc_auc_score(y, v):.6f}')
print('50/50 rank', roc_auc_score(y, pd.Series(res['logreg']).rank() + pd.Series(res['lgb']).rank()))
