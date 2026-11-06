"""Local megastack experiments: public member cache (mega_build.py) + our zoo members, nested CV on SKF(5, rs42).

Usage: python mega_exp.py [--no-dung] [--C 0.1] [--lgb] [--write NAME] [--drop name1,name2] [--only-dung a,b]
"""
import argparse, time, gc, numpy as np, pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--no-dung', action='store_true'); ap.add_argument('--C', type=float, default=0.1)
ap.add_argument('--lgb', action='store_true'); ap.add_argument('--write', default='')
ap.add_argument('--only-dung', default=''); ap.add_argument('--seg', action='store_true'); ap.add_argument('--extra', default='', help='comma list of extra src groups, or all'); ap.add_argument('--threads', type=int, default=8); ap.add_argument('--family', type=float, default=0.0, help='scale member columns by n_family^-alpha'); ap.add_argument('--famtype', action='store_true', help='family = model type parsed from name'); ap.add_argument('--extra2', default='', help='singles | stack | all | comma src list'); ap.add_argument('--drop-src', default=''); ap.add_argument('--raw', action='store_true', help='LR on raw logits (not standardised)'); ap.add_argument('--famw', default='', help='tabpfn,tfm scale e.g. 3,1.5'); ap.add_argument('--gbw', type=float, default=0.0, help='z-score blend weight of ewtherka LightGBM stacker')
args = ap.parse_args()
t0 = time.time()
tr, te = load(); y = tr[TARGET].values; NTR = len(y)
M = pd.read_csv(ART / 'mega_pub_members.csv')
Z = np.load(ART / 'mega_pub_oof.npy'); ZT = np.load(ART / 'mega_pub_test.npy')
names, src = list(M.name), list(M.src)
if args.extra:
    E = pd.read_csv(ART / 'mega_extra_members.csv'); sel = np.ones(len(E), bool) if args.extra == 'all' else E.src.isin(args.extra.split(',')).values
    Z = np.hstack([Z, np.load(ART / 'mega_extra_oof.npy')[:, sel]]); ZT = np.hstack([ZT, np.load(ART / 'mega_extra_test.npy')[:, sel]])
    names += list(E.name[sel]); src += list(E.src[sel])
if args.extra2:
    E = pd.read_csv(ART / 'mega_extra2_members.csv')
    E.loc[E.name.isin(['nj_04', 'nj_05', 'nj_06']), 'src'] = 'stacklevel2'
    if args.extra2 == 'all': sel = np.ones(len(E), bool)
    elif args.extra2 == 'singles': sel = (E.src != 'stacklevel2').values
    elif args.extra2 == 'stack': sel = (E.src == 'stacklevel2').values
    else: sel = E.name.isin(args.extra2.split(',')).values | E.src.isin(args.extra2.split(',')).values
    Z = np.hstack([Z, np.load(ART / 'mega_extra2_oof.npy')[:, sel]]); ZT = np.hstack([ZT, np.load(ART / 'mega_extra2_test.npy')[:, sel]])
    names += list(E.name[sel]); src += list(E.src[sel])
if not args.no_dung:
    zoo = sorted(p.name[:-8] for p in (ART / 'zoo').glob('*_oof.npy'))
    if args.only_dung: zoo = [n for n in zoo if n in args.only_dung.split(',')]
    L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)).astype(np.float32)
    Z = np.hstack([Z, np.column_stack([L(np.load(ART / 'zoo' / f'{n}_oof.npy')) for n in zoo])])
    ZT = np.hstack([ZT, np.column_stack([L(np.load(ART / 'zoo' / f'{n}_test.npy')) for n in zoo])])
    names += ['dung_' + n for n in zoo]; src += ['dung'] * len(zoo)
# de-dup exactly like the public megastack (corr >= 0.99995, keep first)
mu, sd = Z.mean(0), Z.std(0); Z -= mu; Z /= sd; ZT -= mu; ZT /= sd
C_ = (Z.T @ Z) / NTR; keep = []
for i in range(len(names)):
    if all(C_[i, j] < 0.99995 for j in keep): keep.append(i)
Z, ZT = np.ascontiguousarray(Z[:, keep]), np.ascontiguousarray(ZT[:, keep])
names = [names[i] for i in keep]; src = [src[i] for i in keep]; del C_; gc.collect()
print(f'{len(names)} members after de-dup | dung {src.count("dung")} | {time.time()-t0:.0f}s', flush=True)

if args.seg:  # segment one-hots + segment x mean-logit interactions
    cols = ['Class', 'Type of Travel', 'Customer Type']
    S = pd.get_dummies(tr[cols].astype(str)).values.astype(np.float32); ST = pd.get_dummies(te[cols].astype(str)).values.astype(np.float32)
    m_, mt_ = Z.mean(1, keepdims=True), ZT.mean(1, keepdims=True)
    Z = np.hstack([Z, S, S * m_]); ZT = np.hstack([ZT, ST, ST * mt_]); names += [f'seg{i}' for i in range(2 * S.shape[1])]; src += ['seg'] * (2 * S.shape[1])
if args.family:  # family-scaled LR: members of big families get a stronger effective L2 penalty
    if args.famtype:
        import re
        def mtype(n):
            n = n.lower()
            for k in ['tabpfn', 'tabicl', 'tabm', 'realmlp', 'rmlp', 'mlp', 'deepfm', 'cat', 'xgb', 'lgb', 'lgbm', 'glm', 'knn', 'teacher', 'stack', 'blend']:
                if k in n: return {'rmlp': 'realmlp', 'lgbm': 'lgb'}.get(k, k)
            return 'other'
        src = [mtype(n) for n in names]
    fam = pd.Series(src).map(pd.Series(src).value_counts()).values.astype(np.float32)
    sc = fam ** (-args.family); Z *= sc; ZT *= sc
    print('family sizes:', pd.Series(src).value_counts().to_dict())
if args.raw: Z = Z * sd[keep] + mu[keep]; ZT = ZT * sd[keep] + mu[keep]   # back to raw logits
if args.famw:  # ewtherka / goodpjw family-wise L2: up-scale TFM columns = weaker penalty on them
    s_pfn, s_tfm = map(float, args.famw.split(','))
    def fam(n):
        if 'tabpfn' in n or n.startswith('pfn_') or '_pfn_' in n: return s_pfn
        if (n.startswith('cd_') and n not in ('cd_realmlp', 'cd_xgboost')) or 'tabicl' in n or 'tabm' in n: return s_tfm
        return 1.0
    SC = np.array([fam(n) for n in names], np.float32); Z = Z * SC; ZT = ZT * SC
    print('family-scaled cols:', int((SC != 1).sum()))
FOLDS = folds(y)
def report(tag, o):
    print(f'{tag:34s} nested CV {roc_auc_score(y, o):.6f}  folds ' + ' '.join(f'{roc_auc_score(y[b], o[b]):.5f}' for _, b in FOLDS), flush=True)
from joblib import Parallel, delayed
def _fit(a, b):
    return b, LogisticRegression(C=args.C, max_iter=5000, tol=1e-6).fit(Z[a], y[a]).decision_function(Z[b])
lr_oof = np.zeros(NTR)
for b, v in Parallel(n_jobs=5)(delayed(_fit)(a, b) for a, b in FOLDS): lr_oof[b] = v
report(f'LR C={args.C}{" raw" if args.raw else ""}{" fam" + args.famw if args.famw else ""}', lr_oof)
final_oof = lr_oof
if args.write or args.gbw:
    m = LogisticRegression(C=args.C, max_iter=5000, tol=1e-6).fit(Z, y); lr_test = m.decision_function(ZT)
    final_test = lr_test
if args.gbw:  # ewtherka's LightGBM stacker: early stop on an inner 10% split, blend in z-score space
    import lightgbm as lgb
    P = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=args.threads, seed=0)
    def fit_gb(A, ya):
        ai = np.random.default_rng(0).permutation(len(A)); cut = int(len(ai) * 0.9)
        return lgb.train(P, lgb.Dataset(A[ai[:cut]], ya[ai[:cut]]), 3000, valid_sets=[lgb.Dataset(A[ai[cut:]], ya[ai[cut:]])],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
    gb_oof = np.zeros(NTR); its = []
    for a, b in FOLDS:
        g = fit_gb(Z[a], y[a]); its.append(g.best_iteration); gb_oof[b] = g.predict(Z[b], raw_score=True, num_iteration=g.best_iteration)
    report('LGB stacker (ewtherka)', gb_oof)
    zs = lambda v, ref: (v - ref.mean()) / ref.std()
    for w in sorted({0.1, 0.2, 0.3, args.gbw}):
        report(f'{1-w:.2f} LR + {w:.2f} LGB (z)', (1 - w) * zs(lr_oof, lr_oof) + w * zs(gb_oof, gb_oof))
    final_oof = (1 - args.gbw) * zs(lr_oof, lr_oof) + args.gbw * zs(gb_oof, gb_oof)
    g = fit_gb(Z, y); gb_test = g.predict(ZT, raw_score=True, num_iteration=g.best_iteration)
    final_test = (1 - args.gbw) * zs(lr_test, lr_oof) + args.gbw * zs(gb_test, gb_oof)
if args.write:
    np.save(ART / f'lr_oof_{args.write}.npy', final_oof); np.save(ART / f'lr_test_{args.write}.npy', final_test)
    pd.DataFrame({'id': te['id'], TARGET: rankdata(final_test) / len(final_test)}).to_csv(ART / f'submission_{args.write}.csv', index=False); print('wrote', args.write)
print(f'total {time.time()-t0:.0f}s')
