"""Loading and aggregation of the evaluation results (per-sample arrays in ../results/eval/<system>/<scenario>.npz)."""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, '..', 'results', 'eval')
SYSTEMS = ['ieee14', 'ieee30', 'ieee118']
SYS_NAME = {'ieee14': 'IEEE 14-bus', 'ieee30': 'IEEE 30-bus', 'ieee118': 'IEEE 118-bus'}
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# method key -> list of npz prefixes ('seed*' means all seeds of that model)
METHODS = {
    'wls': 'wls', 'rwls': 'rwls', 'lnr': 'lnr', 'ep': 'ep', 'eprob': 'eprob', 'ens': 'ens',
    'dnn': 'nn:dnn_s*', 'pinn': 'nn:pinn_s*', 'dnn_gn': 'gn:dnn_s*', 'pinn_gn': 'gn:pinn_s*', 'hy': 'hy:pinn_s*', 'hyep': 'hyev:pinn_s*', 'hy_dnn': 'hy:dnn_s*',
    'hylam': 'hylam:pinn_s*', 'hylam_s0': 'hylam:pinn_s0', 'hyev': 'hyev:pinn_s*', 'hyev_s0': 'hyev:pinn_s0', 'epev': 'epev', 'hyep0': 'hyep:pinn_s*',
    'hyep_s0': 'hyev:pinn_s0', 'hyep0_s0': 'hyep:pinn_s0', 'hy_s0': 'hy:pinn_s0', 'pinn_s0': 'nn:pinn_s0', 'hy_s01': 'hy:pinn_s0,hy:pinn_s1',
    'hy_dnn_s01': 'hy:dnn_s0,hy:dnn_s1', 'abl_nomeas_s0': 'hy:pinn_nomeas_s0', 'abl_nophys_s0': 'hy:pinn_nophys_s0',
    'abl_fixed_s0': 'hy:pinn_fixed_s0', 'abl_noconf_s0': 'hy:pinn_noconf_s0', 'abl_nodesc_s0': 'hy:pinn_nodesc_s0', 'hy_dnn_s0': 'hy:dnn_s0',
    'hy_q': 'hyq:pinn_s0', 'hy_na': 'hyna:pinn_s0', 'hy_fix': 'hyfix:pinn_s0',
    'abl_fixed': 'hy:pinn_fixed_s*', 'abl_noconf': 'hy:pinn_noconf_s*', 'abl_nodesc': 'hy:pinn_nodesc_s*',
    'abl_nomeas': 'hy:pinn_nomeas_s*', 'abl_nophys': 'hy:pinn_nophys_s*',
    'nn_fixed': 'nn:pinn_fixed_s*', 'nn_noconf': 'nn:pinn_noconf_s*', 'nn_nodesc': 'nn:pinn_nodesc_s*',
    'nn_nomeas': 'nn:pinn_nomeas_s*', 'nn_nophys': 'nn:pinn_nophys_s*',
}
_cache = {}


def load(system, scen):
    key = (system, scen)
    if key not in _cache:
        f = os.path.join(EVAL, system, scen + '.npz')
        _cache[key] = dict(np.load(f)) if os.path.exists(f) else None
    return _cache[key]


def keys_of(d, spec):
    """npz prefixes for a method spec (exact name or prefix with trailing 's*' meaning every seed)."""
    if d is None:
        return []
    if ',' in spec:                                  # explicit list of result prefixes
        return [k for k in spec.split(',') if k + '/mae_v' in d]
    if spec.endswith('s*'):
        pre = spec[:-1]
        ks = sorted({k.split('/')[0] for k in d if k.startswith(pre) and k.endswith('/mae_v')})
        return [k for k in ks if '_L' not in k]
    return [spec] if spec + '/mae_v' in d else []


def per_sample(d, method, field):
    """(seeds, n) array of a per-sample field."""
    ks = keys_of(d, METHODS.get(method, method))
    if not ks:
        return None
    return np.array([d[k + '/' + field] for k in ks], dtype=float)


def mean_over(d, method, field, sel=None):
    """Mean over samples (and seeds); returns (mean, std across seeds, ci95 over samples, n_seeds)."""
    a = per_sample(d, method, field)
    if a is None:
        return (np.nan, np.nan, np.nan, 0)
    if sel is not None:
        if not np.any(sel):
            return (np.nan, np.nan, np.nan, a.shape[0])
        a = a[:, sel]
    per_seed = a.mean(1)
    pooled = a.mean(0)
    ci = 1.96 * pooled.std(ddof=1) / np.sqrt(len(pooled)) if len(pooled) > 1 else np.nan
    return (float(per_seed.mean()), float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan, float(ci), a.shape[0])


def obs_mask(d, which):
    o = d['OBS'].astype(bool)
    return {'all': None, 'obs': o, 'unobs': ~o}[which]


def coverage(d, method, kind='cp', part='v', level=0.9, sel=None):
    """Mean coverage (over samples that have an interval) and mean width at a nominal level."""
    ks = keys_of(d, METHODS.get(method, method))
    if not ks or f'{ks[0]}/{kind}_cov_{part}' not in d:
        return (np.nan, np.nan)
    j = LEVELS.index(level)
    cov, wid = [], []
    for k in ks:
        c, w = d[f'{k}/{kind}_cov_{part}'][:, j], d[f'{k}/{kind}_wid_{part}'][:, j]
        if sel is not None:
            c, w = c[sel], w[sel]
        cov.append(np.nanmean(c) if np.any(np.isfinite(c)) else np.nan)
        wid.append(np.nanmean(w) if np.any(np.isfinite(w)) else np.nan)
    return (float(np.nanmean(cov)), float(np.nanmean(wid)))


def calib_curve(d, method, kind, part, sel=None):
    ks = keys_of(d, METHODS.get(method, method))
    if not ks or f'{ks[0]}/{kind}_cov_{part}' not in d:
        return None
    out = []
    for j in range(len(LEVELS)):
        v = []
        for k in ks:
            c = d[f'{k}/{kind}_cov_{part}'][:, j]
            if sel is not None:
                c = c[sel]
            v.append(np.nanmean(c) if np.any(np.isfinite(c)) else np.nan)
        out.append(np.nanmean(v))
    return np.array(out)


def hp(system):
    return json.load(open(os.path.join(EVAL, f'{system}_hp.json')))
