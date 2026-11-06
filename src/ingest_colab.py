"""Copy Colab results (out_*.zip in Downloads) into artifacts/zoo, sanity-checking shapes and OOF AUC."""
import zipfile, io, numpy as np, json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from common import *
DL = Path.home() / 'Downloads'; ZOO = ART / 'zoo'
y = load()[0][TARGET].values
for z in sorted(DL.glob('out_*.zip')):
    with zipfile.ZipFile(z) as f:
        for n in f.namelist():
            if not n.endswith('_oof.npy'): continue
            name = Path(n).name[:-8]
            oof = np.load(io.BytesIO(f.read(n))); tst = np.load(io.BytesIO(f.read(n.replace('_oof.npy', '_test.npy'))))
            assert oof.shape == (699635,) and tst.shape == (299844,), (name, oof.shape, tst.shape)
            np.save(ZOO / f'{name}_oof.npy', oof); np.save(ZOO / f'{name}_test.npy', tst)
            print(f'{z.name:22s} {name:16s} OOF {roc_auc_score(y, oof):.6f}')
