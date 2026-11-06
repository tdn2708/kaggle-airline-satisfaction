import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace(os.sep, '/')
INPUT = ROOT_DIR + r'/public_oof'
import sys; RUN_BASELINE = '--baseline' in sys.argv
import glob, os, time, json
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
def find_dir(slug):
    hits = [p for p in glob.glob(f'{INPUT}/**/{slug}', recursive=True) if os.path.isdir(p)]
    assert hits, f'input not attached: {slug}'
    return sorted(hits, key=len)[0]

COMP = find_dir('playground-series-s6e10')
tr, te = pd.read_csv(f'{COMP}/train.csv'), pd.read_csv(f'{COMP}/test.csv')
y = tr['satisfaction'].astype(int).values
NTR = len(y)
OOF, TEST, SRC = {}, {}, {}
import psutil, gc
def ram(tag): print(f'[RAM] {tag}: {psutil.Process().memory_info().rss / 1e9:.1f} GB used, {psutil.virtual_memory().available / 1e9:.1f} GB free', flush=True)
def _lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)

def add(name, o, t, src):
    o, t = np.asarray(o, np.float64), np.asarray(t, np.float64)
    if (o.min() < 0) or (o.max() > 1):   # some libraries may store raw scores; squash to (0,1) like the original loader expects
        o, t = 1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-t))
    assert o.shape == (NTR,) and t.shape == (len(te),), (name, o.shape, t.shape)
    assert np.isfinite(o).all() and np.isfinite(t).all(), name
    OOF[name], TEST[name], SRC[name] = _lg(o), _lg(t), src

def by_id(df, ids, col):
    return df.set_index('id')[col].loc[ids].values
# ---------------- kratosyan's 124-member set ----------------
def l_gp():
    d = find_dir('s6e10-tabpfn-route-categories-lb-0-96160')
    go, gt = pd.read_csv(f'{d}/oof_members.csv'), pd.read_csv(f'{d}/test_members.csv')
    assert (go['id'].values == tr['id'].values).all() and (gt['id'].values == te['id'].values).all()
    for c in go.columns[1:]:
        add('gp_' + c, go[c].values, gt[c].values, 'goodpjw P8')
def l_mg():
    d = find_dir('s6e10-oof-library')
    mo, mt = pd.read_parquet(f'{d}/oof_predictions.parquet'), pd.read_parquet(f'{d}/test_predictions.parquet')
    assert (mo['id'].values == tr['id'].values).all() and (mt['id'].values == te['id'].values).all()
    groups = {}
    for c in mo.columns:
        if c in ('id', 'fold', 'satisfaction'):
            continue
        groups.setdefault(c.rsplit('_s', 1)[0] if c.startswith('realmlp') else c, []).append(c)
    for g, cols in groups.items():
        add('mg_' + g, mo[cols].mean(1).values, mt[cols].mean(1).values, 'megayak')
def l_gl():
    d = find_dir('s6e10-golem-oof-library')
    for k in 'abcdefghijklmn':
        add('gl_' + k, np.load(f'{d}/oof_{k}.npy'), np.load(f'{d}/test_{k}.npy'), 'golem')
    for k in ['o_tabpfn35', 'p_lgbm_auxev', 'q_lgbm_auxev_10fold']:   # v3: added to the library on 2026-10-08
        add('gl_' + k, np.load(f'{d}/oof_{k}.npy'), np.load(f'{d}/test_{k}.npy'), 'golem')
def l_ar():
    d = find_dir('s6e10-oof-test-predictions')
    ao, at = pd.read_parquet(f'{d}/oof_predictions.parquet'), pd.read_parquet(f'{d}/test_predictions.parquet')
    assert (ao['id'].values == tr['id'].values).all() and (at['id'].values == te['id'].values).all()
    for c in ['lgb_5f', 'xgb_5f', 'cat_5f', 'realmlp_5f']:
        add('ar_' + c, ao[c].values, at[c].values, 'arhancanli12')
def l_sa():
    d = find_dir('s6e10-stack-oof-predictions')
    for f in sorted(glob.glob(f'{d}/*.npz')):
        z = np.load(f)
        add('sa_' + os.path.basename(f)[:-4], z['oof'], z['test'], 'sachith7')
def l_nj():
    d = find_dir('s6e10-oof')
    add('nj_01', by_id(pd.read_csv(f'{d}/01_blend_oof.csv'), tr['id'], 'satisfaction'),
        by_id(pd.read_csv(f'{d}/01_submission.csv'), te['id'], 'satisfaction'), 'najiama')
def l_pfn():
    for nm, slug in [('pfn_samanyu_route_og', 's6e10-tabpfn-route-og'), ('pfn_samanyu_seeds', 's6e10-tabpfn-seeds'),
                     ('pfn_hemingweb_raw', 's6e10-exp01-tabpfn35-full-context')]:
        d = find_dir(slug)
        add(nm, by_id(pd.read_csv(f'{d}/oof_exp01_tabpfn35.csv'), tr['id'], 'pred'),
            by_id(pd.read_csv(f'{d}/test_exp01_tabpfn35.csv'), te['id'], 'satisfaction'), 'TabPFN outputs')
def l_mi():
    d = find_dir('s6e10-analysis-first-equality-blocks-stack')
    mo2 = pd.read_parquet(f'{d}/oof_predictions.parquet').set_index('id').loc[tr['id']]
    mt2 = pd.read_parquet(f'{d}/test_predictions.parquet').set_index('id').loc[te['id']]
    for c in mo2.columns:
        add('mi_' + c, mo2[c].values, mt2[c].values, 'mitudru')
def l_wx():
    for slug, c in [('s6e10-catboost-rating-crosses-cv-0-96080', 'rating_segments'),
                    ('s6e10-lightgbm-context-stats-cv-0-96065', 'context_numeric')]:
        d = find_dir(slug)
        add('wx_' + c, by_id(pd.read_parquet(f'{d}/oof_predictions.parquet'), tr['id'], c),
            by_id(pd.read_parquet(f'{d}/test_predictions.parquet'), te['id'], c), 'wangxintong111')
def l_th():
    for slug, tag in [('s6e10-xgb-realmlp-catboost-lb-0-96123', 'thisray'), ('s6e10-honest-oof-stack', 'megayak honest')]:
        d = find_dir(slug)
        o = pd.read_csv(f'{d}/oof_members.csv').set_index('id').loc[tr['id']]
        t = pd.read_csv(f'{d}/test_members.csv').set_index('id').loc[te['id']]
        for c in t.columns:
            add(f'{tag.split()[0][:2]}{"h" if "honest" in tag else ""}_{c}', o[c].values, t[c].values, tag)
def l_bp():
    d = find_dir('s6e10-busyaprime-five-members-oof')
    for f in sorted(glob.glob(f'{d}/bp_*.npz')):
        z = np.load(f)
        add(os.path.basename(f)[:-4], z['oof'], z['test'], 'busyaprime')

# ---------------- new members (not in kratosyan v3) ----------------
def l_ab():  # abdullahsafwan333: 15-fold x 3-seed TabPFN-3.5 variants + CatBoost/XGB (15 folds)
    d = find_dir('oof-dataset')
    for m in ['tabpfn_catfd10', 'tabpfn_raw', 'tabpfn_te_cnt', 'cat_v3', 'xgb_v3']:
        add('ab_' + m, by_id(pd.read_csv(f'{d}/{m}_oof.csv'), tr['id'], 'pred'),
            by_id(pd.read_csv(f'{d}/{m}_test.csv'), te['id'], 'pred'), 'abdullah ' + ('pfn' if 'tabpfn' in m else 'gbdt'))
def l_kx():  # kratosyan 10-fold route + original-teacher + aux members
    d = find_dir('s6e10-xgboost-realmlp-route-id-teacher')
    for f in sorted(glob.glob(f'{d}/**/*_route_ogaux.npz', recursive=True)):
        z = np.load(f)
        add('kx_' + os.path.basename(f)[:-4], z['oof'], z['test'], 'kratosyan10f')
def l_hg():
    d = find_dir('single-catboost-eda-ps6e10')
    add('hg_cat', np.load(glob.glob(f'{d}/**/oof_train.npy', recursive=True)[0]),
        np.load(glob.glob(f'{d}/**/oof_test.npy', recursive=True)[0]), 'hermengardo')
def l_dp():
    d = find_dir('s6e10-numeric-keys-catboost-ctr')
    o = pd.read_csv(glob.glob(f'{d}/**/oof_predictions.csv', recursive=True)[0])
    tp = np.zeros(len(te)); n = 0
    for f in sorted(glob.glob(f'{d}/**/catboost_numeric_ctr_fold*.npz', recursive=True)):
        z = np.load(f)
        assert (z['test_ids'] == te['id'].values).all()
        tp += z['test_pred']; n += 1
    assert n == 5, n
    add('dp_catctr', by_id(o, tr['id'], 'catboost_numeric_ctr'), tp / n, 'denpugovkin')
def l_yk():
    d = find_dir('ps-s6-e10-realmlp-pytabkit')
    add('yk_realmlp', by_id(pd.read_csv(f'{d}/oof_preds.csv'), tr['id'], 'satisfaction'),
        by_id(pd.read_csv(f'{d}/submission.csv'), te['id'], 'satisfaction'), 'yekenot')


# Sadam Torres artifacts (38 members + 2 original-data teacher predictions)
def l_st():
    d = find_dir('ps-s6e10-airline-satisfaction-artifacts')
    for f in sorted(glob.glob(f'{d}/**/preds_*.npz', recursive=True)):
        z = np.load(f)
        add('st_' + os.path.basename(f)[6:-4], z['oof'], z['test'], 'sadamtorres')
    for f in ['orig_cat_pred', 'orig_model_pred']:
        z = np.load(glob.glob(f'{d}/**/{f}.npz', recursive=True)[0])
        add('st_' + f, z['train'], z['test'], 'orig teacher')
# Aman Atar original-prior stack v5 (5 base members)
def l_am():
    d = find_dir('s6e10-the-original-prior-stack-v5')
    z = np.load(glob.glob(f'{d}/**/oof_artifacts.npz', recursive=True)[0])
    for k in ['lgbm_champion', 'lgbm_te', 'lgbm_te_rows', 'xgb_gpu', 'catboost_gpu']:
        add('am_' + k, z['oof_' + k], z['test_' + k], 'amanatar')
# Ahmet Kagan Koral (5 members)
def l_kk():
    d = find_dir('s6e10-encodings-catboost-realmlp-lb-0-961')
    o, t = pd.read_csv(f'{d}/oof_members.csv'), pd.read_csv(f'{d}/test_members.csv')
    assert (o['id'].values == tr['id'].values).all() and (t['id'].values == te['id'].values).all()
    for c in o.columns[1:]:
        add('kk_' + c, o[c].values, t[c].values, 'kagankoral')
# stack-level OOFs: busyaprime's stack (LB 0.96169) and najiama blends 02/03
def l_stacklevel():
    d = find_dir('s6e10-route-or-distance-both-lb-0-96169')
    z = np.load(glob.glob(f'{d}/**/oof_stacks.npz', recursive=True)[0])
    add('bp_stack53', z['oof53'], z['test53'], 'stacklevel')
    d = find_dir('s6e10-oof')
    for k in ['02', '03']:
        add('nj_' + k, by_id(pd.read_csv(f'{d}/{k}_blend_oof.csv'), tr['id'], 'satisfaction'),
            by_id(pd.read_csv(f'{d}/{k}_submission.csv'), te['id'], 'satisfaction'), 'stacklevel')

# Chris Deotte: 9 tabular foundation models + tuned XGBoost / RealMLP (SKF5 rs42, train/test row order)
def l_cd():
    d = find_dir('s6e10-tfm-oof-and-test-predictions')
    for f in sorted(glob.glob(f'{d}/**/oof_*.npy', recursive=True)):
        m = os.path.basename(f)[4:-4]
        add('cd_' + m, np.load(f), np.load(f.replace('oof_', 'test_')), 'cdeotte tfm')

# Every source is optional: if an author re-runs or removes a file, that library is skipped (and reported) instead of
# breaking the notebook. Kernel outputs are attached at their latest version, so the member count can change over time.

# samanyu1808: our 202-member library (23 own GBDT/RealMLP/TabPFN members + forge, cdeotte-124 extras, ...), stored as logits
def l_ms():
    d = find_dir('s6e10-megastack-members')
    m = pd.read_csv(f'{d}/members.csv'); O, T = np.load(f'{d}/oof_members.npy'), np.load(f'{d}/test_members.npy')
    for i, n in enumerate(m.name):
        add('ms_' + n, 1 / (1 + np.exp(-O[:, i].astype(float))), 1 / (1 + np.exp(-T[:, i].astype(float))), 'samanyu1808')

SKIPPED = []
for fn in [l_gp, l_mg, l_gl, l_ar, l_sa, l_nj, l_pfn, l_mi, l_wx, l_th, l_bp, l_ab, l_kx, l_hg, l_dp, l_yk,
           l_st, l_am, l_kk, l_stacklevel, l_cd, l_ms]:
    before = dict(OOF)
    try:
        fn()
    except Exception as e:
        for k in set(OOF) - set(before):   # roll back a partially loaded library
            OOF.pop(k); TEST.pop(k); SRC.pop(k)
        SKIPPED.append(fn.__name__)
        print(f'skipped {fn.__name__}: {type(e).__name__}: {e}')
names = list(OOF)
print(len(names), 'members |', pd.Series(SRC).value_counts().to_dict())
print(f'loaded in {time.time() - T0:.0f}s; skipped libraries: {SKIPPED}')


# cache public member logits (float32) so stack experiments do not reload ~25 libraries
Z = np.empty((NTR, len(names)), np.float32); ZT = np.empty((len(te), len(names)), np.float32)
for i, n in enumerate(names): Z[:, i] = OOF.pop(n); ZT[:, i] = TEST.pop(n)
np.save(ROOT_DIR + r'/artifacts/mega_pub_oof.npy', Z); np.save(ROOT_DIR + r'/artifacts/mega_pub_test.npy', ZT)
pd.DataFrame({'name': names, 'src': [SRC[n] for n in names]}).to_csv(ROOT_DIR + r'/artifacts/mega_pub_members.csv', index=False)
ram('cached'); print('cached', Z.shape, ZT.shape)
