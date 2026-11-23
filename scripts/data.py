"""
Operating scenarios and measurement corruption for the H-PINN state-estimation study.

States come from AC power flows (PYPOWER Newton-Raphson) of the IEEE 14/30/118-bus cases with random load level, per-bus load
noise, PV and wind output.  Measurements are generated with the exact measurement functions of grid.py and corrupted by
Gaussian or contaminated noise, group-wise missing data (random or area outage), systematic bias and network-parameter error.
All random draws are seeded; datasets are cached in ../data as .npz files.
"""
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import Grid

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
os.makedirs(DATA, exist_ok=True)

ID_RANGE = dict(load=(0.80, 1.15), pv=(0.0, 0.8), wind=(0.0, 0.8))                       # training (in-distribution)
SHIFT_RANGE = dict(load=((0.62, 0.80), (1.15, 1.30)), pv=(0.8, 1.0), wind=(0.75, 1.0))   # unseen operating conditions
LOAD_NOISE = {'uni': 0.10, 'id': 0.05}                                                    # per-bus load factor std
SHARE_NOISE = 0.10                                                                       # per-snapshot dispatch-share noise
CONTAM = dict(pi=0.10, kappa=10.0)                                                       # contaminated Gaussian noise
AREA_FRAC = 0.25                                                                         # share of buses in an outage area
DEG_FACTOR = 3.0                                                                         # accuracy-class factor of degraded devices
N_TRAIN_REGIMES = 5                                                                      # regime 5 is held out for the shift test
N_WORKERS = 4

SPEC0 = dict(op='id', eta=1.0, contam=0.0, miss=0.0, area=0.0, bias=0.0, bias_frac=0.05, param=0.0, deg=0.1)


def spec(**kw):
    d = dict(SPEC0)
    d.update(kw)
    return d


# training / validation / calibration mixture: every uncertainty type appears at random
MIX = spec(eta=('logu', 0.7, 2.0), contam=0.25, miss=('mix', 0.6, ('u', 0.0, 0.5)), area=0.10,
           bias=('mix', 0.25, ('u', 3.0, 10.0)), param=('mix', 0.3, ('u', 0.01, 0.05)))


def draw(v, rng):
    if isinstance(v, tuple):
        if v[0] == 'u':
            return float(rng.uniform(v[1], v[2]))
        if v[0] == 'logu':
            return float(np.exp(rng.uniform(np.log(v[1]), np.log(v[2]))))
        if v[0] == 'mix':
            return draw(v[2], rng) if rng.random() < v[1] else 0.0
    return v


_GC = {}


def get_grid(name):
    if name not in _GC:
        _GC[name] = Grid(name)
    return _GC[name]


def groups(g):
    """Measurement groups (one device / RTU channel): V, injection pair, flow pair, PMU pair; zero injections have id -1."""
    if hasattr(g, '_grp'):
        return g._grp, g._ghost
    kind = {'V': 'V', 'Pi': 'I', 'Qi': 'I', 'Pf': 'F', 'Qf': 'F', 'PVm': 'P', 'PVa': 'P'}
    ids, grp, host = {}, np.full(g.M, -1, int), []
    for i, (t, l) in enumerate(zip(g.mtype, g.mloc)):
        if t.startswith('ZI'):
            continue
        key = (kind[t], int(l))
        if key not in ids:
            ids[key] = len(ids)
            host.append(int(g.mhost[i]))
        grp[i] = ids[key]
    g._grp, g._ghost = grp, np.array(host)
    return g._grp, g._ghost


# ------------------------------------------------------------------------------------------------ operating states
def _state_job(args):
    name, seed, op, spar = args
    g = get_grid(name)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        if op in ('id', 'uni'):
            ls = rng.uniform(*ID_RANGE['load'])
            apv, aw = rng.uniform(*ID_RANGE['pv']), rng.uniform(*ID_RANGE['wind'])
        else:
            lo, hi = SHIFT_RANGE['load'][rng.integers(2)]
            ls = rng.uniform(lo, hi)
            apv, aw = rng.uniform(*SHIFT_RANGE['pv']), rng.uniform(*SHIFT_RANGE['wind'])
        reg = None if op == 'uni' else (int(rng.integers(N_TRAIN_REGIMES)) if op == 'id' else N_TRAIN_REGIMES)
        bf = np.clip(1 + (LOAD_NOISE['uni'] if op == 'uni' else LOAD_NOISE['id']) * rng.standard_normal(g.N), 0.6, 1.4)
        sn = None if reg is None else np.clip(1 + SHARE_NOISE * rng.standard_normal(g.share.shape[1]), 0.7, 1.3)
        xp = np.clip(1 + spar * rng.standard_normal(g.L), 0.5, 1.5) if spar > 0 else None
        Va, Vm, ok, br = g.solve_pf(ls, bf, apv, aw, xpert=xp, regime=reg, share_noise=sn)
        if ok and 0.85 < Vm.min() and Vm.max() < 1.2:
            return g.x_from(Va, Vm), g.true_meas(Va, Vm, br), np.array([ls, apv, aw, spar, -1 if reg is None else reg])
    raise RuntimeError('power flow failed repeatedly')


def get_states(name, K, op, spar_spec, seed, tag=None, workers=N_WORKERS):
    """K power-flow states (x_true, clean measurements, meta). spar_spec: parameter-error std (number or draw tuple)."""
    f = os.path.join(DATA, name, f'states_{tag}.npz') if tag else None
    if f and os.path.exists(f):
        d = np.load(f)
        return d['X'], d['ZT'], d['META']
    rng = np.random.default_rng(seed)
    spars = [draw(spar_spec, rng) for _ in range(K)]
    seeds = rng.integers(0, 2 ** 31 - 1, size=K)
    tasks = [(name, int(seeds[k]), op, spars[k]) for k in range(K)]
    if workers > 1:
        with Pool(workers) as p:
            out = p.map(_state_job, tasks, chunksize=max(1, K // (workers * 8)))
    else:
        out = [_state_job(t) for t in tasks]
    X = np.array([o[0] for o in out]); ZT = np.array([o[1] for o in out]); META = np.array([o[2] for o in out])
    if f:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.savez_compressed(f, X=X, ZT=ZT, META=META)
    return X, ZT, META


# ------------------------------------------------------------------------------------------------ measurement corruption
def build_meas(g, X, ZT, sp, seed):
    rng = np.random.default_rng(seed)
    K, M = ZT.shape
    grp, ghost = groups(g)
    ng = len(ghost)
    inzi = grp < 0
    gi = np.maximum(grp, 0)
    Z = np.zeros((K, M)); MASK = np.ones((K, M), bool); SIG = np.zeros((K, M)); BAD = np.zeros((K, M), bool)
    INFO = np.zeros((K, 6))                    # eta, miss rate, area flag, bias beta, n biased groups, contaminated flag
    for k in range(K):
        eta, miss, beta = draw(sp['eta'], rng), draw(sp['miss'], rng), draw(sp['bias'], rng)
        area = rng.random() < sp['area']
        contam = rng.random() < sp['contam']
        cm = np.where(inzi, 1.0, np.where(rng.random(ng) < sp['deg'], DEG_FACTOR, 1.0)[gi])
        sig = g.sigma * cm                                              # declared standard deviation
        eps = rng.standard_normal(M)
        if contam:
            eps = np.where(rng.random(M) < CONTAM['pi'], CONTAM['kappa'] * eps, eps)
        noise = eta * sig * eps
        noise[g.is_zi] = 0.0
        z = ZT[k] + noise
        drop = rng.random(ng) < miss
        if area:
            ar = g.neighbours(int(rng.integers(g.N)), max(2, int(round(AREA_FRAC * g.N))))
            ins = np.zeros(g.N, bool); ins[ar] = True
            drop |= ins[ghost]
        mask = np.where(inzi, True, ~drop[gi])
        nb = 0
        if beta > 0:
            avail = np.where(~drop)[0]
            nb = min(len(avail), max(1, int(round(sp['bias_frac'] * len(avail)))))
            pick = rng.choice(avail, nb, replace=False)
            sgn = np.where(rng.random(ng) < 0.5, -1.0, 1.0)
            sel = np.zeros(ng, bool); sel[pick] = True
            hit = (~inzi) & sel[gi]
            z = z + np.where(hit, sgn[gi] * beta * sig, 0.0)
            BAD[k] = hit & mask
        Z[k] = np.where(mask, z, 0.0); MASK[k] = mask; SIG[k] = sig
        INFO[k] = [eta, miss if not area else miss, float(area), beta, nb, float(contam)]
    return dict(Z=Z, MASK=MASK, SIG=SIG, BAD=BAD, INFO=INFO)


def make_set(name, K, sp, seed, tag, states=None, workers=N_WORKERS, with_obs=False):
    """Full dataset (states + corrupted measurements, optionally observability flags); cached."""
    f = os.path.join(DATA, name, f'set_{tag}.npz')
    if os.path.exists(f):
        return dict(np.load(f))
    g = get_grid(name)
    if states is None:
        X, ZT, META = get_states(name, K, sp['op'], sp['param'], seed, tag=None, workers=workers)
    else:
        X, ZT, META = states
    d = build_meas(g, X, ZT, sp, seed + 1)
    d.update(X=X, META=META)
    if with_obs:
        add_observability(name, d, workers)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez_compressed(f, **d)
    return d


def observability(g, x, mask, sig, tol=1e-7):
    """Numerical observability at the true state: smallest/largest singular value of the row-normalised Jacobian."""
    H = g.H(x)
    rows = np.where(mask)[0]
    if len(rows) < g.nx:
        return False, 0.0
    A = H[rows] / np.linalg.norm(H[rows], axis=1, keepdims=True).clip(1e-12)
    s = np.linalg.svd(A, compute_uv=False)
    return bool(s[-1] / s[0] > tol), float(s[-1] / s[0])


def _obs_job(args):
    name, x, mask, sig = args
    return observability(get_grid(name), x, mask, sig)


def add_observability(name, d, workers=N_WORKERS):
    tasks = [(name, d['X'][k], d['MASK'][k], d['SIG'][k]) for k in range(len(d['X']))]
    with Pool(workers) as p:
        out = p.map(_obs_job, tasks, chunksize=8)
    d['OBS'] = np.array([o[0] for o in out]); d['SMIN'] = np.array([o[1] for o in out])
    return d


# ------------------------------------------------------------------------------------------------ scenario catalogue
def scenarios():
    S = {}
    S['nominal'] = spec()
    for e in (0.5, 2.0, 3.0, 4.0):
        S[f'noise{e:g}'] = spec(eta=e)
    S['contam'] = spec(contam=1.0)
    for p in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        S[f'miss{int(round(p * 100))}'] = spec(miss=p)
    S['area'] = spec(area=1.0)
    for b in (3.0, 6.0, 10.0):
        S[f'bias{b:g}'] = spec(bias=b)
    for s in (0.02, 0.05, 0.10):
        S[f'param{int(s * 100)}'] = spec(param=s)
    S['shift'] = spec(op='shift')
    S['shift_miss30'] = spec(op='shift', miss=0.3)
    S['mix'] = dict(MIX)
    return S


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'ieee14'
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    g = get_grid(name)
    t0 = time.time()
    X, ZT, META = get_states(name, K, 'id', 0.0, 1, tag=None, workers=4)
    print('states', X.shape, f'{time.time() - t0:.1f}s', 'V range', X[:, g.N - 1:].min(), X[:, g.N - 1:].max())
    d = build_meas(g, X, ZT, MIX, 2)
    r = (d['Z'] - ZT) / d['SIG']
    m = d['MASK'] & ~d['BAD'] & ~g.is_zi[None, :]
    print('normalised residual std (clean, kept):', r[m].std(), '| missing fraction', 1 - d['MASK'].mean(),
          '| bias flagged fraction', d['BAD'].mean(), '| info mean', d['INFO'].mean(0))
    t0 = time.time()
    d['OBS'] = np.array([observability(g, d['X'][k] if 'X' in d else X[k], d['MASK'][k], d['SIG'][k])[0] for k in range(min(K, 100))])
    print('observable fraction (first 100):', d['OBS'].mean(), f'{time.time() - t0:.1f}s')
