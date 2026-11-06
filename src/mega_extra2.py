"""Cache OOF members published 2026-10-11..18 (public_oof3/) as float32 logits: artifacts/mega_extra2_*.

Single models: golem v6 (21, oof_wide.parquet), goodpjw2008 extra members, kagankoral ablation members,
koumeimaki route/TFM/NN members, dmitriykolmikov all-categorical CatBoost, najiama blends 04-07.
Stack-level OOFs (golem stack247, goodpjw stack245) get src 'stacklevel2' so they can be tested separately.
"""
import glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import *

P = ROOT / 'public_oof3'
tr, te = load(); y = tr[TARGET].values
OOF, TEST, SRC = {}, {}, {}
def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)
def add(name, o, t, src):
    o, t = np.asarray(o, np.float64).ravel(), np.asarray(t, np.float64).ravel()
    assert o.shape == (len(tr),) and t.shape == (len(te),), (name, o.shape, t.shape)
    assert np.isfinite(o).all() and np.isfinite(t).all(), name
    if o.min() < 0 or o.max() > 1: o, t = 1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-t))
    OOF[name], TEST[name], SRC[name] = lg(o), lg(t), src
def by_id(df, ids, col):
    return df.set_index('id')[col].loc[ids].values
def pairs(d, prefix, src, skip=()):
    for f in sorted(d.glob('oof_*.npy')):
        k = f.name[4:-4]
        if k in skip or not (d / f'test_{k}.npy').exists(): continue
        add(prefix + k, np.load(f), np.load(d / f'test_{k}.npy'), src)

# golem v6: 21 single models in wide parquet files (+ stack247 kept apart)
d = P / 's6e10-golem-oof-library'
ow, tw = pd.read_parquet(d / 'oof_wide.parquet'), pd.read_parquet(d / 'test_wide.parquet')
ow, tw = ow.set_index('id').loc[tr['id']], tw.set_index('id').loc[te['id']]
for c in [c for c in ow.columns if c not in ('fold',)]:
    add('gl6_' + c, ow[c].values, tw[c].values, 'golem6')
add('gl6_stack247', np.load(d / 'stack247_oof.npy'), np.load(d / 'stack247_test.npy'), 'stacklevel2')
# goodpjw2008 extra members (+ stack245 kept apart)
d = P / 's6e10-goodpjw2008-extra-members-oof'
pairs(d, 'gpx_', 'goodpjw extra', skip=('stack245',))
if (d / 'oof_stack245.npy').exists():
    add('gpx_stack245', np.load(d / 'oof_stack245.npy'), np.load(d / 'test_stack245.npy'), 'stacklevel2')
pairs(P / 's6e10-ablation-members-oof', 'kka_', 'kagankoral ablation')
# koumeimaki: rows in train_ids_folds.csv / test_ids.csv order
d = P / 's6e10-route-tfm-nn-oof'
tid, eid = pd.read_csv(d / 'train_ids_folds.csv')['id'].values, pd.read_csv(d / 'test_ids.csv')['id'].values
assert (tid == tr['id'].values).all() and (eid == te['id'].values).all()
pairs(d, 'km_', 'koumeimaki')
d = P / 's6e10-catboost-all-categorical-oof-and-test'
add('dk_cb_allcat', np.load(d / 'oof_cb_cat.npy'), np.load(d / 'test_cb_cat.npy'), 'dmitriykolmikov')
d = P / 's6e10-oof'
for k in ['04', '05', '06']:
    add(f'nj_{k}', by_id(pd.read_csv(d / f'{k}_blend_oof.csv'), tr['id'], 'satisfaction'),
        by_id(pd.read_csv(d / f'{k}_submission.csv'), te['id'], 'satisfaction'), 'najiama new')
add('nj_07_xgb', by_id(pd.read_csv(d / '07_XGBoost_oof.csv'), tr['id'], 'oof_pred'),
    by_id(pd.read_csv(d / '07_submission.csv'), te['id'], 'satisfaction'), 'najiama new')

names = list(OOF)
for n in names: print(f'{n:40s} {SRC[n]:20s} {roc_auc_score(y, OOF[n]):.6f}')
np.save(ART / 'mega_extra2_oof.npy', np.column_stack([OOF[n] for n in names]))
np.save(ART / 'mega_extra2_test.npy', np.column_stack([TEST[n] for n in names]))
pd.DataFrame({'name': names, 'src': [SRC[n] for n in names]}).to_csv(ART / 'mega_extra2_members.csv', index=False)
print(len(names), 'extra2 members cached')
