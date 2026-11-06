import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace(os.sep, '/')
"""Download every input of the public 'S6E10 Megastack' Kaggle notebook into /content/input/<slug>/ via the Kaggle API.

The list of inputs is read from the notebook's own metadata (kaggle kernels pull -m), so nothing is guessed.
"""
import csv, io, json, os, subprocess, sys, zipfile, glob

INPUT = ROOT_DIR + r'/public_oof'; os.makedirs(INPUT, exist_ok=True)
COMP = 'playground-series-s6e10'

def sh(args, check=False):
    p = subprocess.run(['kaggle'] + args, capture_output=True, text=True)
    if check and p.returncode != 0: raise RuntimeError(f'kaggle {" ".join(args)}\n{p.stdout}\n{p.stderr}')
    return p

# 1) find the megastack notebook
ref = 'samanyu1808/s6e10-megastack-please-upvote'
if not ref:
    p = sh(['kernels', 'list', '--competition', COMP, '-s', 'megastack', '--page-size', '100', '-v'], check=True)
    rows = list(csv.DictReader(io.StringIO(p.stdout)))
    refs = [r['ref'] for r in rows if 'megastack-please-upvote' in r['ref']] or [r['ref'] for r in rows if 'megastack' in r['ref']]
    print('megastack candidates:', refs[:10])
    assert refs, 'no megastack notebook found'
    ref = refs[0]
print('using notebook:', ref)
os.makedirs(ROOT_DIR + r'/public_oof/_meta', exist_ok=True)
sh(['kernels', 'pull', ref, '-p', ROOT_DIR + r'/public_oof/_meta', '-m'], check=True)
meta = json.load(open(glob.glob(ROOT_DIR + r'/public_oof/_meta/*metadata*.json')[0]))
print('datasets:', len(meta.get('dataset_sources', [])), '| notebook outputs:', len(meta.get('kernel_sources', [])))

# 2) competition data
d = f'{INPUT}/{COMP}'
if not os.path.exists(f'{d}/train.csv'):
    os.makedirs(d, exist_ok=True)
    sh(['competitions', 'download', '-c', COMP, '-p', d], check=True)
    for z in glob.glob(f'{d}/*.zip'):
        zipfile.ZipFile(z).extractall(d); os.remove(z)
print('competition files:', os.listdir(d))

# 3) datasets + notebook outputs (failures are reported, not fatal: the megastack loaders skip missing libraries)
failed = []
for kind, srcs in [('dataset', meta.get('dataset_sources', [])), ('kernel', meta.get('kernel_sources', []))]:
    for s in srcs:
        slug = s.split('/')[-1]; d = f'{INPUT}/{slug}'
        if os.path.isdir(d) and os.listdir(d):
            print('have', s); continue
        os.makedirs(d, exist_ok=True)
        args = ['datasets', 'download', '-d', s, '-p', d, '--unzip', '-q'] if kind == 'dataset' else ['kernels', 'output', s, '-p', d, '-o']
        p = sh(args)
        n = sum(len(f) for _, _, f in os.walk(d))
        print(f'{"OK " if p.returncode == 0 and n else "ERR"} {kind:7s} {s}  ({n} files)', flush=True)
        if p.returncode != 0 or not n: failed.append(s); print('   ', (p.stderr or p.stdout)[-300:])
print('\nfailed sources:', failed)
pass
