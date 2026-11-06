"""Train one model config with 5-fold CV; save OOF + test predictions to artifacts/zoo/<name>_{oof,test}.npy.

Usage: python train.py <name> [--threads 16] [--folds 5]
"""
import argparse, time, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import *; from feats import *

ap = argparse.ArgumentParser()
ap.add_argument('name'); ap.add_argument('--threads', type=int, default=16); ap.add_argument('--folds', type=int, default=N_FOLDS)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--params', default='', help='json overrides of model params')
ap.add_argument('--tag', default='')
args = ap.parse_args()
ZOO = ART / 'zoo'; ZOO.mkdir(parents=True, exist_ok=True)

tr, te = load(); y = tr[TARGET].values; o = load_orig()
R_tr, R_te = route_stats(tr, te, o)

def featset(fs):
    Xtr, Xte = [base(tr)], [base(te)]
    for c in CATS: Xte[0][c] = pd.Categorical(Xte[0][c], categories=Xtr[0][c].cat.categories)
    if 'r' in fs: Xtr.append(R_tr); Xte.append(R_te)
    if 'o' in fs: Xtr.append(orig_te(tr, o)); Xte.append(orig_te(te, o))
    if 'rc' in fs:
        a, b = route_seg_stats(tr, te, o, 'Class'); Xtr.append(a); Xte.append(b)
    if 'x' in fs: Xtr.append(extra(tr)); Xte.append(extra(te))
    if 'e' in fs:
        a, b = route_orig_extra(tr, te, o); Xtr.append(a); Xte.append(b)
    if 'i' in fs:
        a, b = aux_inflight(tr, te, o, R_tr, R_te)
        Xtr.append(pd.DataFrame({'aux_inflight': a}, index=tr.index)); Xte.append(pd.DataFrame({'aux_inflight': b}, index=te.index))
    if 't' in fs:
        a, b = teacher(tr, te, o, R_tr, R_te)
        Xtr.append(pd.DataFrame({'teacher': a}, index=tr.index)); Xte.append(pd.DataFrame({'teacher': b}, index=te.index))
    return pd.concat(Xtr, axis=1), pd.concat(Xte, axis=1)

CONFIGS = {
    'lgb_r': dict(algo='lgb', fs=['r'], p=dict(learning_rate=0.02, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                            reg_alpha=5, reg_lambda=5, min_child_samples=100)),
    'lgb_rot': dict(algo='lgb', fs=['r', 'o', 'rc'], te=True, p=dict(learning_rate=0.02, num_leaves=127, max_depth=10, subsample=0.7, subsample_freq=1,
                                            colsample_bytree=0.4, reg_alpha=10, reg_lambda=1, min_child_samples=200, max_bin=1023)),
    'lgb_rt': dict(algo='lgb', fs=['r', 't'], p=dict(learning_rate=0.02, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                            reg_alpha=5, reg_lambda=5, min_child_samples=100)),
    'lgb_rte': dict(algo='lgb', fs=['r', 't', 'e'], p=dict(learning_rate=0.02, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                            reg_alpha=5, reg_lambda=5, min_child_samples=100)),
    'lgb_rti': dict(algo='lgb', fs=['r', 't', 'i'], p=dict(learning_rate=0.02, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                            reg_alpha=5, reg_lambda=5, min_child_samples=100)),
    'xgb_rt': dict(algo='xgb', fs=['r', 't'], p=dict(learning_rate=0.02, max_depth=8, subsample=0.8, colsample_bytree=0.5, min_child_weight=20,
                                            reg_alpha=2, reg_lambda=5, max_bin=512)),
    'cat_rt': dict(algo='cat', fs=['r', 't', 'o'], p=dict(learning_rate=0.06, depth=7, l2_leaf_reg=3, grow_policy='Lossguide', max_leaves=64,
                                            border_count=254), catnum=True),
    'lgb_t': dict(algo='lgb', fs=['t'], p=dict(learning_rate=0.02, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                            reg_alpha=5, reg_lambda=5, min_child_samples=100)),
    'xgb_t': dict(algo='xgb', fs=['t'], p=dict(learning_rate=0.02, max_depth=8, subsample=0.8, colsample_bytree=0.5, min_child_weight=20,
                                            reg_alpha=2, reg_lambda=5, max_bin=512)),
    'lgb_cat': dict(algo='lgb', fs=['r', 't'], ascat=True, p=dict(learning_rate=0.02, num_leaves=63, subsample=0.8, subsample_freq=1,
                                            colsample_bytree=0.5, reg_alpha=5, reg_lambda=5, min_child_samples=100, cat_smooth=50,
                                            min_data_per_group=200, max_cat_threshold=64, cat_l2=10)),
    'lgb_ctx': dict(algo='lgb', fs=['r', 't', 'f'], te='ctx', p=dict(learning_rate=0.02, num_leaves=63, subsample=0.8, subsample_freq=1,
                                            colsample_bytree=0.4, reg_alpha=5, reg_lambda=5, min_child_samples=100)),
    'xgb_ctx': dict(algo='xgb', fs=['r', 't', 'f'], te='ctx', p=dict(learning_rate=0.02, max_depth=8, subsample=0.8, colsample_bytree=0.4,
                                            min_child_weight=20, reg_alpha=2, reg_lambda=5, max_bin=512)),
    'cat_ctx': dict(algo='cat', fs=['r', 't', 'f'], te='ctx', p=dict(learning_rate=0.06, depth=8, l2_leaf_reg=5, border_count=254)),
    'lgb_ctx2': dict(algo='lgb', fs=['r', 't', 'f'], te='ctx', k2=True, p=dict(learning_rate=0.02, num_leaves=63, subsample=0.8, subsample_freq=1,
                                            colsample_bytree=0.3, reg_alpha=5, reg_lambda=5, min_child_samples=100)),
    'xgb_r': dict(algo='xgb', fs=['r'], p=dict(learning_rate=0.02, max_depth=8, subsample=0.8, colsample_bytree=0.5, min_child_weight=20,
                                            reg_alpha=2, reg_lambda=5, max_bin=512)),
    'xgb_rox': dict(algo='xgb', fs=['r', 'o', 'x'], p=dict(learning_rate=0.02, max_depth=0, grow_policy='lossguide', max_leaves=96, subsample=0.7,
                                            colsample_bytree=0.4, min_child_weight=50, reg_lambda=10, max_bin=256)),
    'cat_r': dict(algo='cat', fs=['r'], p=dict(learning_rate=0.06, depth=8, l2_leaf_reg=5, border_count=254)),
    'cat_ro': dict(algo='cat', fs=['r', 'o'], p=dict(learning_rate=0.06, depth=7, l2_leaf_reg=3, grow_policy='Lossguide', max_leaves=64,
                                            border_count=254), catnum=True),
}
cfg = CONFIGS[args.name]
if args.params: cfg['p'].update(json.loads(args.params))
X, Xt = featset(cfg['fs'])
if cfg.get('te') == 'ctx': KC_tr, KC_te = (combo_keys2(tr), combo_keys2(te)) if cfg.get('k2') else (combo_keys(tr), combo_keys(te))
if 'f' in cfg['fs']:
    a_, b_ = freq_enc(tr, te); X = pd.concat([X, a_], axis=1); Xt = pd.concat([Xt, b_], axis=1)
if cfg.get('ascat'):  # route id + ratings as native categoricals (different split structure)
    for c in ['Flight Distance'] + SERV:
        cats = pd.Index(sorted(set(X[c]) | set(Xt[c])))
        X[c] = pd.Categorical(X[c], categories=cats); Xt[c] = pd.Categorical(Xt[c], categories=cats)
print(args.name, X.shape, flush=True)

def fit_predict(Xa, ya, Xb, yb, Xt):
    a, p = cfg['algo'], dict(cfg['p'])
    if a == 'lgb':
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=30000, verbose=-1, n_jobs=args.threads, random_state=args.seed, **p)
        m.fit(Xa, ya, eval_X=(Xb,), eval_y=(yb,), eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
        return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1], m.best_iteration_
    if a == 'xgb':
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=30000, tree_method='hist', enable_categorical=True, eval_metric='auc', early_stopping_rounds=300,
                              n_jobs=args.threads, random_state=args.seed, max_cat_to_onehot=4, **p)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1], m.best_iteration
    if a == 'cat':
        from catboost import CatBoostClassifier, Pool
        cats = list(CATS)
        conv = lambda d: d.assign(**{c: d[c].astype(str) for c in cats})
        Xa, Xb, Xt = conv(Xa), conv(Xb), conv(Xt)
        if cfg.get('catnum'):  # route id + ratings also as categorical copies -> CTR features
            extra_c = ['Flight Distance'] + SERV[:6]
            for d in (Xa, Xb, Xt):
                for c in extra_c: d[f'c_{c}'] = d[c].astype(str)
            cats += [f'c_{c}' for c in extra_c]
        m = CatBoostClassifier(iterations=30000, eval_metric='AUC', od_type='Iter', od_wait=300, thread_count=args.threads,
                               random_seed=args.seed, verbose=0, **p)
        m.fit(Pool(Xa, ya, cat_features=cats), eval_set=Pool(Xb, yb, cat_features=cats), use_best_model=True)
        return m.predict_proba(Pool(Xb, cat_features=cats))[:, 1], m.predict_proba(Pool(Xt, cat_features=cats))[:, 1], m.get_best_iteration()

oof = np.zeros(len(y)); tst = np.zeros(len(te))
for k, (a, b) in enumerate(folds(y)):
    if k >= args.folds: break
    t = time.time()
    Xa, Xb, Xtk = X.iloc[a], X.iloc[b], Xt
    if cfg.get('te') == 'ctx':
        src = pd.concat([tr[CATS + SERV + NUMS], KC_tr], axis=1); srct = pd.concat([te[CATS + SERV + NUMS], KC_te], axis=1)
        A, (B, T) = fold_te(src.iloc[a], y[a], [src.iloc[b], srct], list(src.columns))
        Xa, Xb, Xtk = pd.concat([Xa, A], axis=1), pd.concat([Xb, B], axis=1), pd.concat([Xt, T], axis=1)
    elif cfg.get('te'):
        A, (B, T) = fold_te(tr.iloc[a], y[a], [tr.iloc[b], te], CATS + SERV + NUMS)
        Xa, Xb, Xtk = pd.concat([Xa, A], axis=1), pd.concat([Xb, B], axis=1), pd.concat([Xt, T], axis=1)
    pb, pt, it = fit_predict(Xa, y[a], Xb, y[b], Xtk)
    oof[b] = pb; tst += pt / N_FOLDS
    print(f'{args.name} fold {k} it {it} auc {roc_auc_score(y[b], pb):.6f} {time.time()-t:.0f}s', flush=True)
if args.folds < N_FOLDS: raise SystemExit
auc = roc_auc_score(y, oof)
print(f'{args.name} OOF AUC {auc:.6f}', flush=True)
tag = args.tag or (args.name if args.seed == 0 else f'{args.name}_s{args.seed}')
np.save(ZOO / f'{tag}_oof.npy', oof); np.save(ZOO / f'{tag}_test.npy', tst)
with open(ZOO / 'scores.jsonl', 'a') as f: f.write(json.dumps({'name': tag, 'oof_auc': auc}) + '\n')
