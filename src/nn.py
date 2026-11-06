"""Neural models from pytabkit (RealMLP / TabM) on base + route features. Run with .venv_nn.

Usage: ..\\.venv_nn\\Scripts\\python nn.py realmlp [--threads 8]
"""
import argparse, time, json, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from common import *; from feats import *

ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--folds', type=int, default=N_FOLDS)
args = ap.parse_args(); torch.set_num_threads(args.threads)
ZOO = ART / 'zoo'; ZOO.mkdir(parents=True, exist_ok=True)

tr, te = load(); y = tr[TARGET].values; o = load_orig()
R_tr, R_te = route_stats(tr, te, o)
def mk(df, R):
    X = pd.concat([df[CATS + SERV + NUMS], R], axis=1)
    X['Arrival Delay in Minutes'] = X['Arrival Delay in Minutes'].fillna(X['Departure Delay in Minutes'])
    for c in ['Departure Delay in Minutes', 'Arrival Delay in Minutes']: X[c] = np.log1p(X[c])
    for c in CATS: X[c] = X[c].astype('category')
    X['route_id'] = df['Flight Distance'].astype(str).astype('category')  # embedding of the route
    num = [c for c in X.columns if c not in CATS + ['route_id']]
    X[num] = X[num].fillna(0).astype('float32')
    return X
X, Xt = mk(tr, R_tr), mk(te, R_te)
if args.name.endswith('_t'):
    ta, tb = teacher(tr, te, o, R_tr, R_te); X['teacher'] = ta.astype('float32'); Xt['teacher'] = tb.astype('float32')
for c in CATS + ['route_id']:
    cats = pd.Index(sorted(set(X[c].astype(str)) | set(Xt[c].astype(str))))
    X[c] = pd.Categorical(X[c].astype(str), categories=cats); Xt[c] = pd.Categorical(Xt[c].astype(str), categories=cats)

def model():
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
    common = dict(device='cpu', random_state=0, n_threads=args.threads, val_metric_name='1-auc_ovr', verbosity=0)
    if args.name.startswith('realmlp'):
        return RealMLP_TD_Classifier(n_epochs=12, batch_size=1024, use_ls=False, **common)
    if args.name.startswith('tabm'):
        return TabM_D_Classifier(n_epochs=12, batch_size=1024, **common)
    raise ValueError(args.name)

oof = np.zeros(len(y)); tst = np.zeros(len(te))
for k, (a, b) in enumerate(folds(y)):
    if k >= args.folds: break
    t = time.time(); m = model()
    m.fit(X.iloc[a], y[a], X.iloc[b], y[b])
    oof[b] = m.predict_proba(X.iloc[b])[:, 1]; tst += m.predict_proba(Xt)[:, 1] / N_FOLDS
    print(f'{args.name} fold {k} auc {roc_auc_score(y[b], oof[b]):.6f} {time.time()-t:.0f}s', flush=True)
if args.folds == N_FOLDS:
    auc = roc_auc_score(y, oof); print(f'{args.name} OOF AUC {auc:.6f}', flush=True)
    np.save(ZOO / f'{args.name}_oof.npy', oof); np.save(ZOO / f'{args.name}_test.npy', tst)
    with open(ZOO / 'scores.jsonl', 'a') as f: f.write(json.dumps({'name': args.name, 'oof_auc': auc}) + '\n')
