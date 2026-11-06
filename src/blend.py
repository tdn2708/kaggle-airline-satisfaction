"""Blend zoo models. Compares rank-mean, hill-climbing weights and logistic stacking on logits (nested CV).

Usage: python blend.py [model names...]   (default: every *_oof.npy in artifacts/zoo)
Writes artifacts/submission_blend.csv using the method with the best nested-CV AUC.
"""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from common import *

ZOO = ART / 'zoo'
HILL = '--hill' in sys.argv; sys.argv = [a for a in sys.argv if a != '--hill']
names = sys.argv[1:] or sorted(p.name[:-8] for p in ZOO.glob('*_oof.npy'))
tr, te = load(); y = tr[TARGET].values
O = np.column_stack([np.load(ZOO / f'{n}_oof.npy') for n in names])
T = np.column_stack([np.load(ZOO / f'{n}_test.npy') for n in names])
for n, c in zip(names, O.T): print(f'{n:28s} {roc_auc_score(y, c):.6f}')

def L(p): return logit(np.clip(p, 1e-6, 1 - 1e-6))
LO, LT = L(O), L(T)
# rank-normalise per model so models with different calibration mix fairly in hill climbing
RO = np.column_stack([rankdata(c) / len(c) for c in O.T]); RT = np.column_stack([rankdata(c) / len(c) for c in T.T])

def hill(Z, yy, iters=40):
    w = np.zeros(Z.shape[1]); best = np.argmax([roc_auc_score(yy, c) for c in Z.T]); w[best] = 1; cur = Z[:, best].copy()
    s = roc_auc_score(yy, cur)
    for _ in range(iters):
        cand = [(roc_auc_score(yy, (cur * w.sum() + Z[:, j]) / (w.sum() + 1)), j) for j in range(Z.shape[1])]
        sc, j = max(cand)
        if sc <= s + 1e-7: break
        cur = (cur * w.sum() + Z[:, j]) / (w.sum() + 1); w[j] += 1; s = sc
    return w / w.sum()

def stack_fit(Z, yy, C=1.0):
    return LogisticRegression(C=C, max_iter=2000).fit(Z, yy)

outer = StratifiedKFold(5, shuffle=True, random_state=123)
res = {'mean_rank': np.zeros(len(y)), 'stack': np.zeros(len(y))}
if HILL: res['hill'] = np.zeros(len(y))
for a, b in outer.split(O, y):
    res['mean_rank'][b] = RO[b].mean(1)
    if HILL: w = hill(RO[a], y[a]); res['hill'][b] = RO[b] @ w
    m = stack_fit(LO[a], y[a]); res['stack'][b] = m.decision_function(LO[b])
scores = {k: roc_auc_score(y, v) for k, v in res.items()}   # note: per-fold scale differences can only hurt pooled AUC
fold_scores = {k: np.mean([roc_auc_score(y[b], v[b]) for _, b in outer.split(O, y)]) for k, v in res.items()}
for k in res: print(f'{k:10s} nested pooled {scores[k]:.6f}  mean-fold {fold_scores[k]:.6f}')

best = max(fold_scores, key=fold_scores.get)
if best == 'mean_rank': pt = RT.mean(1); wdesc = 'equal'
elif best == 'hill':
    w = hill(RO, y); pt = RT @ w; wdesc = dict(zip(names, np.round(w, 3)))
else:
    m = stack_fit(LO, y); pt = expit(m.decision_function(LT)); wdesc = dict(zip(names, np.round(m.coef_[0], 3)))
print('chosen', best, wdesc)
pt = rankdata(pt) / len(pt)  # AUC only needs ranks
sub = pd.DataFrame({'id': te['id'], TARGET: pt})
sub.to_csv(ART / 'submission_blend.csv', index=False); print('wrote', ART / 'submission_blend.csv', sub.shape)
