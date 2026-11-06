"""Feature builders. Original-data features use only the public original dataset (allowed by rules)."""
import numpy as np, pandas as pd
from common import *

def load_orig():
    o = pd.read_csv(DATA / 'original/original.csv')
    o[TARGET] = (o['__FeatEng_target__'] == 'satisfied').astype(int)
    o['Arrival Delay in Minutes'] = o['Arrival Delay in Minutes'].astype(float)
    return o

def _key(df, cols):
    return pd.util.hash_pandas_object(df[cols].astype(str), index=False).values

ORIG_KEYS = {
    'all17': CATS + SERV,
    'serv13': SERV,
    'all17_age': CATS + SERV + ['Age'],
    'cats_age': CATS + ['Age'],
    'dist': ['Flight Distance'],
    'cats_dist': CATS + ['Flight Distance'],
}

def orig_te(df, o, prior=20):
    """Smoothed original-dataset target mean + count for several column-combination keys."""
    out = {}; p = o[TARGET].mean()
    for name, cols in ORIG_KEYS.items():
        ko = _key(o, cols); g = pd.DataFrame({'k': ko, 'y': o[TARGET].values}).groupby('k')['y'].agg(['sum', 'count'])
        kd = pd.Series(_key(df, cols))
        s = kd.map(g['sum']).fillna(0).values; n = kd.map(g['count']).fillna(0).values
        out[f'o_{name}_te'] = (s + prior * p) / (n + prior)
        out[f'o_{name}_n'] = n
    return pd.DataFrame(out, index=df.index)

def base(df):
    X = df[CATS + SERV + NUMS].copy()
    for c in CATS: X[c] = X[c].astype('category')
    return X

def extra(df):
    X = pd.DataFrame(index=df.index)
    s = df[SERV]
    X['serv_mean'] = s.mean(1); X['serv_std'] = s.std(1); X['n_zero'] = (s == 0).sum(1); X['n_five'] = (s == 5).sum(1)
    X['delay_diff'] = df['Arrival Delay in Minutes'] - df['Departure Delay in Minutes']
    X['arr_na'] = df['Arrival Delay in Minutes'].isna().astype(int)
    return X

# ---------------- route (Flight Distance) features ----------------
RD = 'Flight Distance'

def route_stats(tr, te, o):
    """Label-free per-route aggregates over train+test+original, plus original satisfaction rate per route."""
    allx = pd.concat([tr.drop(columns=[TARGET, 'id']), te.drop(columns=['id']), o[tr.columns.drop([TARGET, 'id'])]], ignore_index=True)
    allx['_biz'] = (allx['Class'] == 'Business').astype(float); allx['_eco'] = (allx['Class'] == 'Eco').astype(float)
    allx['_btr'] = (allx['Type of Travel'] == 'Business travel').astype(float)
    allx['_loy'] = (allx['Customer Type'] == 'Loyal Customer').astype(float)
    allx['_fem'] = (allx['Gender'] == 'Female').astype(float)
    allx['_dly'] = (allx['Departure Delay in Minutes'] > 0).astype(float)
    agg_cols = SERV + ['Age', '_biz', '_eco', '_btr', '_loy', '_fem', '_dly']
    g = allx.groupby(RD)[agg_cols].mean().add_prefix('r_mean_')
    g['r_cnt'] = allx.groupby(RD).size()
    g['r_age_std'] = allx.groupby(RD)['Age'].std()
    p = o[TARGET].mean(); og = o.groupby(RD)[TARGET].agg(['sum', 'count'])
    g['r_orig_te'] = ((og['sum'] + 10 * p) / (og['count'] + 10)).reindex(g.index).fillna(p)
    g['r_orig_n'] = og['count'].reindex(g.index).fillna(0)
    def apply(df):
        R = g.reindex(df[RD].values).set_index(df.index)
        for c in SERV + ['Age']: R[f'r_dev_{c}'] = df[c] - R[f'r_mean_{c}']
        return R
    return apply(tr), apply(te)

def fold_te(Xa, ya, Xb_list, cols, smooth='auto'):
    """sklearn TargetEncoder: cross-fitted on the training part, full-fit for the held-out parts."""
    from sklearn.preprocessing import TargetEncoder
    enc = TargetEncoder(target_type='binary', smooth=smooth, cv=5, shuffle=True, random_state=0)
    A = enc.fit_transform(Xa[cols].astype(str), ya)
    Bs = [enc.transform(Xb[cols].astype(str)) for Xb in Xb_list]
    names = [f'te_{c}' for c in cols]
    return pd.DataFrame(A, index=Xa.index, columns=names), [pd.DataFrame(B, index=Xb.index, columns=names) for B, Xb in zip(Bs, Xb_list)]

def route_seg_stats(tr, te, o, seg='Class'):
    """Per (route, segment) label-free rating means over train+test+original."""
    cols = tr.columns.drop([TARGET, 'id'])
    allx = pd.concat([tr[cols], te[cols], o[cols]], ignore_index=True)
    key = [RD, seg]
    agg = ['Inflight wifi service', 'Online boarding', 'Inflight entertainment', 'Seat comfort', 'Ease of Online booking', 'Age']
    g = allx.groupby(key)[agg].mean().add_prefix(f'rs_{seg[:4]}_')
    g[f'rs_{seg[:4]}_cnt'] = allx.groupby(key).size()
    og = o.groupby(key)[TARGET].agg(['sum', 'count']); p = o[TARGET].mean()
    g[f'rs_{seg[:4]}_orig_te'] = ((og['sum'] + 10 * p) / (og['count'] + 10)).reindex(g.index)
    def apply(df):
        idx = pd.MultiIndex.from_frame(df[key])
        return g.reindex(idx).set_index(df.index)
    return apply(tr), apply(te)

def teacher(tr, te, o, R_tr, R_te, threads=4):
    """LightGBM trained only on the original dataset (base + route stats); its logit is a feature for comp rows.
    Uses no competition labels, so it is leak-free for CV. Cached in artifacts/teacher.npz."""
    f = ART / 'teacher.npz'
    if f.exists():
        z = np.load(f); return z['tr'], z['te']
    import lightgbm as lgb
    _, R_o = route_stats(tr, o.assign(id=0)[tr.columns.drop(TARGET)], o)
    Xo = pd.concat([base(o), R_o], axis=1)
    Xs = []
    for df, R in ((tr, R_tr), (te, R_te)):
        X = pd.concat([base(df), R], axis=1)
        for c in CATS: X[c] = pd.Categorical(X[c], categories=Xo[c].cat.categories)
        Xs.append(X)
    m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                           min_child_samples=30, verbose=-1, n_jobs=threads).fit(Xo, o[TARGET].values)
    a, b = (m.predict(X, raw_score=True) for X in Xs)
    np.savez(f, tr=a, te=b); return a, b

def route_orig_extra(tr, te, o):
    """Per-route means from the ORIGINAL rows only: hidden 'Inflight service' column and delays."""
    g = o.groupby(RD).agg(r_o_inflight=('Inflight service', 'mean'), r_o_dep=('Departure Delay in Minutes', 'mean'),
                          r_o_arr=('Arrival Delay in Minutes', 'mean'))
    allx = pd.concat([tr, te])
    g2 = allx.groupby(RD).agg(r_dep_mean=('Departure Delay in Minutes', 'mean'), r_dly_pos=('Departure Delay in Minutes', lambda s: (s > 0).mean()))
    G = g.join(g2, how='outer')
    f = lambda df: G.reindex(df[RD].values).set_index(df.index)
    return f(tr), f(te)

def aux_inflight(tr, te, o, R_tr, R_te, threads=8):
    """Predict the dropped 'Inflight service' rating from the other columns (model fit on original rows)."""
    f = ART / 'aux_inflight.npz'
    if f.exists():
        z = np.load(f); return z['tr'], z['te']
    import lightgbm as lgb
    _, R_o = route_stats(tr, o.assign(id=0)[tr.columns.drop(TARGET)], o)
    Xo = pd.concat([base(o), R_o], axis=1)
    Xs = []
    for df, R in ((tr, R_tr), (te, R_te)):
        X = pd.concat([base(df), R], axis=1)
        for c in CATS: X[c] = pd.Categorical(X[c], categories=Xo[c].cat.categories)
        Xs.append(X)
    m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                          verbose=-1, n_jobs=threads).fit(Xo, o['Inflight service'].values)
    a, b = (m.predict(X) for X in Xs)
    np.savez(f, tr=a, te=b); return a, b

def combo_keys(df):
    """String keys for conditional target encoding (rating x segment, segment combos, route x segment)."""
    K = pd.DataFrame(index=df.index)
    seg3 = df['Type of Travel'].str[:3] + df['Customer Type'].str[:3] + df['Class']
    K['k_seg3'] = seg3; K['k_seg3_age'] = seg3 + df['Age'].astype(str)
    K['k_route_cls'] = df[RD].astype(str) + df['Class']; K['k_route_tt'] = df[RD].astype(str) + df['Type of Travel']
    for c in SERV:
        K[f'k_{c[:12]}_tc'] = df[c].astype(str) + df['Type of Travel'] + df['Class']
        K[f'k_{c[:12]}_ct'] = df[c].astype(str) + df['Customer Type']
    K['k_wifi_board'] = df['Inflight wifi service'].astype(str) + df['Online boarding'].astype(str) + seg3
    K['k_ent_seat'] = df['Inflight entertainment'].astype(str) + df['Seat comfort'].astype(str) + df['Class']
    return K

def freq_enc(tr, te):
    allx = pd.concat([tr, te]); out = []
    for df in (tr, te):
        F = pd.DataFrame(index=df.index)
        for c in CATS + SERV + NUMS: F[f'f_{c}'] = df[c].map(allx[c].value_counts()).fillna(0).values / len(allx)
        out.append(F)
    return out

TOP6 = ['Online boarding', 'Inflight wifi service', 'Inflight entertainment', 'Seat comfort', 'Ease of Online booking', 'Leg room service']

def combo_keys2(df):
    """combo_keys + rating pairs x travel type, rating x class x customer type, age bin x segment, route x class x travel."""
    K = combo_keys(df)
    from itertools import combinations
    for a, b in combinations(TOP6, 2):
        K[f'k2_{a[:8]}_{b[:8]}'] = df[a].astype(str) + df[b].astype(str) + df['Type of Travel']
    for c in SERV:
        K[f'k2_{c[:12]}_cc'] = df[c].astype(str) + df['Class'] + df['Customer Type']
    ab = (df['Age'] // 5).astype(str)
    K['k2_age_seg'] = ab + df['Type of Travel'] + df['Class'] + df['Customer Type'] + df['Gender']
    K['k2_route_ct'] = df[RD].astype(str) + df['Class'] + df['Type of Travel']
    return K
