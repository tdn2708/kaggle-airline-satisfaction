import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'; ART = ROOT / 'artifacts'
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
SERV = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness']
NUMS = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
TARGET = 'satisfaction'
N_FOLDS = 5; SEED = 42

def load():
    tr = pd.read_csv(DATA / 'train.csv'); te = pd.read_csv(DATA / 'test.csv')
    tr[TARGET] = tr[TARGET].astype(int)
    return tr, te

def folds(y):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
