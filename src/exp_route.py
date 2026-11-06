import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import *; from feats import *
variant = sys.argv[1]; nf = int(sys.argv[2]) if len(sys.argv) > 2 else 5
tr, te = load(); y = tr[TARGET].values; o = load_orig()
parts = [base(tr)]
if 'r' in variant.split('_'): parts.append(route_stats(tr, te, o)[0])
if 'o' in variant.split('_'): parts.append(orig_te(tr, o))
if 'rc' in variant.split('_'): parts.append(route_seg_stats(tr, te, o, 'Class')[0])
if 'rt' in variant.split('_'): parts.append(route_seg_stats(tr, te, o, 'Type of Travel')[0])
MB = 4095 if 'mb' in variant.split('_') else 255
LR = 0.02 if 'lr' in variant.split('_') else 0.05
X = pd.concat(parts, axis=1)
TE_COLS = CATS + SERV + NUMS
for k, (a, b) in enumerate(folds(y)):
    if k >= nf: break
    Xa, Xb = X.iloc[a], X.iloc[b]
    if 't' in variant.split('_'):
        A, (B,) = fold_te(tr.iloc[a], y[a], [tr.iloc[b]], TE_COLS)
        Xa = pd.concat([Xa, A], axis=1); Xb = pd.concat([Xb, B], axis=1)
    REG = dict(reg_alpha=5, reg_lambda=5, min_child_samples=100) if "reg" in variant.split("_") else {}
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=LR, max_bin=MB, num_leaves=63, subsample=0.8, subsample_freq=1, **REG,
                           colsample_bytree=0.5, verbose=-1, n_jobs=16)
    m.fit(Xa, y[a], eval_X=(Xb,), eval_y=(y[b],), eval_metric='auc', callbacks=[lgb.early_stopping(200, verbose=False)])
    p = m.predict_proba(Xb)[:, 1]
    print(variant, k, m.best_iteration_, round(roc_auc_score(y[b], p), 6), flush=True)
