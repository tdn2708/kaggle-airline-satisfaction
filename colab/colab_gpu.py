"""GPU model zoo for Kaggle s6e10 (Airline Satisfaction). Called from airline_gpu.ipynb.

Reads train_feats.parquet / test_feats.parquet (base + route stats + teacher, folds fixed in __fold)
and writes <name>_oof.npy / <name>_test.npy into OUT. Models already in OUT are skipped (resumable).
"""
import os, sys, time, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

DATA = sys.argv[1]; OUT = sys.argv[2]; MODELS = sys.argv[3].split()
os.makedirs(OUT, exist_ok=True)
X = pd.read_parquet(f'{DATA}/train_feats.parquet'); Xt = pd.read_parquet(f'{DATA}/test_feats.parquet')
DEV = os.environ.get('DEV', 'cuda')
if os.environ.get('SMOKE'):  # quick local check on a small subsample
    X = X.sample(20000, random_state=0); Xt = Xt.iloc[:3000]
    try:
        import torch; torch.set_num_threads(4)
    except ImportError: pass
y = X.pop('__y').values; fold = X.pop('__fold').values; Xt.pop('__id')
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
NF = 1 if os.environ.get('SMOKE') else int(fold.max()) + 1

def nn_frame(D, teacher=True):
    D = D.copy()
    if not teacher: D = D.drop(columns=['teacher'])
    D['Arrival Delay in Minutes'] = D['Arrival Delay in Minutes'].fillna(D['Departure Delay in Minutes'])
    for c in ['Departure Delay in Minutes', 'Arrival Delay in Minutes']: D[c] = np.log1p(D[c])
    D['route_id'] = D['Flight Distance'].astype(int).astype(str)
    num = [c for c in D.columns if c not in CATS + ['route_id']]
    D[num] = D[num].fillna(0).astype('float32')
    return D

def as_cat(A, B, cols):
    for c in cols:
        cats = pd.Index(sorted(set(A[c].astype(str)) | set(B[c].astype(str))))
        A[c] = pd.Categorical(A[c].astype(str), categories=cats); B[c] = pd.Categorical(B[c].astype(str), categories=cats)
    return A, B

def run(name, make_fit):
    if os.path.exists(f'{OUT}/{name}_test.npy'):
        print(name, 'exists, skip'); return
    oof = np.zeros(len(y)); tst = np.zeros(len(Xt))
    for k in range(NF):
        a, b = np.where(fold != k)[0], np.where(fold == k)[0]; t = time.time()
        pb, pt = make_fit(a, b, k)
        oof[b] = pb; tst += pt / NF
        print(f'{name} fold {k} auc {roc_auc_score(y[b], pb):.6f} {time.time()-t:.0f}s', flush=True)
    auc = roc_auc_score(y, oof); print(f'{name} OOF AUC {auc:.6f}', flush=True)
    np.save(f'{OUT}/{name}_oof.npy', oof); np.save(f'{OUT}/{name}_test.npy', tst)
    with open(f'{OUT}/scores.jsonl', 'a') as f: f.write(json.dumps({'name': name, 'oof_auc': auc}) + '\n')

def pytabkit_model(kind, seed, teacher=True, **kw):
    N, Nt = as_cat(nn_frame(X, teacher), nn_frame(Xt, teacher), CATS + ['route_id'])
    def fit(a, b, k):
        from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
        common = dict(device=DEV, random_state=seed + k, val_metric_name='1-auc_ovr', verbosity=0)
        if os.environ.get('SMOKE'): kw.update(n_epochs=2)
        m = (RealMLP_TD_Classifier if kind == 'realmlp' else TabM_D_Classifier)(**common, **kw)
        m.fit(N.iloc[a], y[a], N.iloc[b], y[b])
        return m.predict_proba(N.iloc[b])[:, 1], m.predict_proba(Nt)[:, 1]
    return fit

def cat_gpu(depth, seed):
    from catboost import CatBoostClassifier, Pool
    C, Ct = X.copy(), Xt.copy(); cats = list(CATS)
    for c in ['Flight Distance', 'Online boarding', 'Inflight wifi service', 'Inflight entertainment', 'Seat comfort']:
        C[f'c_{c}'] = C[c].astype(int).astype(str); Ct[f'c_{c}'] = Ct[c].astype(int).astype(str); cats.append(f'c_{c}')
    def fit(a, b, k):
        m = CatBoostClassifier(iterations=50 if os.environ.get('SMOKE') else 20000, learning_rate=0.05, depth=depth, l2_leaf_reg=5, border_count=254, task_type='GPU' if DEV == 'cuda' else 'CPU',
                               eval_metric='AUC', od_type='Iter', od_wait=400, random_seed=seed, verbose=0)
        m.fit(Pool(C.iloc[a], y[a], cat_features=cats), eval_set=Pool(C.iloc[b], y[b], cat_features=cats), use_best_model=True)
        return m.predict_proba(Pool(C.iloc[b], cat_features=cats))[:, 1], m.predict_proba(Pool(Ct, cat_features=cats))[:, 1]
    return fit

def xgb_gpu(depth, seed):
    import xgboost as xgb
    G, Gt = X.copy(), Xt.copy()
    for c in CATS: G[c] = G[c].astype('category'); Gt[c] = pd.Categorical(Gt[c], categories=G[c].cat.categories)
    def fit(a, b, k):
        m = xgb.XGBClassifier(n_estimators=50 if os.environ.get('SMOKE') else 30000, learning_rate=0.01, max_depth=depth, subsample=0.8, colsample_bytree=0.4,
                              min_child_weight=30, reg_alpha=3, reg_lambda=5, max_bin=1024, tree_method='hist', device=DEV,
                              enable_categorical=True, eval_metric='auc', early_stopping_rounds=500, random_state=seed)
        m.fit(G.iloc[a], y[a], eval_set=[(G.iloc[b], y[b])], verbose=False)
        return m.predict_proba(G.iloc[b])[:, 1], m.predict_proba(Gt)[:, 1]
    return fit

ZOO = {
    'tabm':        lambda: pytabkit_model('tabm', 0, n_epochs=30, batch_size=1024),
    'tabm_s1':     lambda: pytabkit_model('tabm', 100, n_epochs=30, batch_size=1024),
    'realmlp_g':   lambda: pytabkit_model('realmlp', 0, n_epochs=30, batch_size=512),
    'realmlp_g_s1': lambda: pytabkit_model('realmlp', 100, n_epochs=30, batch_size=512),
    'realmlp_g_nt': lambda: pytabkit_model('realmlp', 200, teacher=False, n_epochs=30, batch_size=512),
    'tabm_nt':     lambda: pytabkit_model('tabm', 300, teacher=False, n_epochs=30, batch_size=1024),
    'realmlp_big': lambda: pytabkit_model('realmlp', 400, n_epochs=40, batch_size=512, hidden_sizes=[512, 512, 512], p_drop=0.15),
    'cat_gpu_d10': lambda: cat_gpu(10, 0),
    'cat_gpu_d8':  lambda: cat_gpu(8, 1),
    'xgb_gpu_d7':  lambda: xgb_gpu(7, 1),
    'xgb_gpu_d9':  lambda: xgb_gpu(9, 0),
}
for name in MODELS:
    t = time.time(); run(name, ZOO[name]()); print(f'== {name} done in {(time.time()-t)/60:.1f} min', flush=True)
