"""Cache extra public OOF members (published after the megastack v4) as float32 logits: artifacts/mega_extra_*.

Sources (public_oof2/): amanatar forge pairs, goodpjw2008 TabPFN/TabICL members, kozykappa source-transfer,
cdeotte 124-pair library (if downloaded). Members are checked for shape, id order and finiteness.
"""
import glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import *

P = ROOT / 'public_oof2'
tr, te = load(); y = tr[TARGET].values
OOF, TEST, SRC = {}, {}, {}
def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)
def add(name, o, t, src):
    o, t = np.asarray(o, np.float64), np.asarray(t, np.float64)
    assert o.shape == (len(tr),) and t.shape == (len(te),), (name, o.shape, t.shape)
    assert np.isfinite(o).all() and np.isfinite(t).all(), name
    if o.min() < 0 or o.max() > 1: o, t = 1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-t))
    OOF[name], TEST[name], SRC[name] = lg(o), lg(t), src

d = P / 's6e10-forge-own-pairs'
for f in sorted(d.glob('oof_*.npy')):
    k = f.name[4:-4]; add('fg_' + k, np.load(f), np.load(d / f'test_{k}.npy'), 'forge')
d = P / 's6e10-tabpfn-member'
for f in sorted(d.glob('oof_*.csv')):
    k = f.name[4:-4]; o = pd.read_csv(f); t = pd.read_csv(d / f'test_{k}.csv')
    assert (o['id'].values == tr['id'].values).all() and (t['id'].values == te['id'].values).all()
    add('gpm_' + k, o['pred'].values, t['pred'].values, 'goodpjw member')
d = P / 's6e10-source-transfer-foundation-model-oof'
add('kz_source_transfer', np.load(d / 'source_transfer_oof.npy'), np.load(d / 'source_transfer_test.npy'), 'kozykappa')
d = P / 'cdeotte124'
for f in sorted(glob.glob(str(d / '**' / 'oof_*.npy'), recursive=True)):
    k = os.path.basename(f)[4:-4]; tf = f.replace('oof_', 'test_')
    if os.path.exists(tf): add('cd124_' + k, np.load(f), np.load(tf), 'cdeotte124')

names = list(OOF)
for n in names: print(f'{n:40s} {SRC[n]:16s} {roc_auc_score(y, OOF[n]):.6f}')
np.save(ART / 'mega_extra_oof.npy', np.column_stack([OOF[n] for n in names]))
np.save(ART / 'mega_extra_test.npy', np.column_stack([TEST[n] for n in names]))
pd.DataFrame({'name': names, 'src': [SRC[n] for n in names]}).to_csv(ART / 'mega_extra_members.csv', index=False)
print(len(names), 'extra members cached')
