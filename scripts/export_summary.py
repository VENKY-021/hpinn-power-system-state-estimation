"""
Compact result summary for traceability: mean errors, failure rate, iterations, coverage and safeguard factors of every estimator in
every scenario of every system (from the per-snapshot files results/eval/<system>/<scenario>.npz).  Output ../results/summary.json.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis as A
from data import scenarios

METHODS = ['wls', 'rwls', 'lnr', 'ep', 'eprob', 'epev', 'dnn', 'pinn', 'dnn_gn', 'hy', 'hyep0', 'hyep', 'ens']
out = {'description': 'mean over test snapshots (and training seeds for networks); mae_v in p.u., mae_a in degrees; hyep = H-PINN (proposed, with safeguard), hyep0 = without safeguard, epev = EP-RWLS with safeguard',
       'systems': {}}
for system in A.SYSTEMS + ['u_ieee30']:
    out['systems'][system] = {}
    for sc in scenarios():
        d = A.load(system, sc)
        if d is None:
            continue
        rec = {'n_snapshots': int(len(d['OBS'])), 'observable_share': float(d['OBS'].mean()), 'methods': {}}
        for m in METHODS:
            v = A.mean_over(d, m, 'mae_v')
            if not np.isfinite(v[0]):
                continue
            a = A.mean_over(d, m, 'mae_a')
            r = {'mae_v': v[0], 'mae_v_sd_seeds': None if not np.isfinite(v[1]) else v[1], 'mae_a': a[0]}
            for f in ('ok', 'iters'):
                try:
                    r['fail_rate' if f == 'ok' else 'iters'] = float(1 - A.mean_over(d, m, f)[0]) if f == 'ok' else float(A.mean_over(d, m, f)[0])
                except KeyError:
                    pass
            for sel in ('obs', 'unobs'):
                mask = A.obs_mask(d, sel)
                n_s = A.per_sample(d, m, 'mae_v').shape[1]
                if mask.any() and len(mask) == n_s:
                    r[f'mae_v_{sel}'] = A.mean_over(d, m, 'mae_v', mask)[0]
                    r[f'mae_a_{sel}'] = A.mean_over(d, m, 'mae_a', mask)[0]
            for kind in ('raw', 'cp'):
                cv, wv = A.coverage(d, m, kind, 'v', 0.9)
                ca, wa = A.coverage(d, m, kind, 'a', 0.9)
                if np.isfinite(cv):
                    r[f'{kind}_cov90_v'], r[f'{kind}_width90_v'], r[f'{kind}_cov90_a'], r[f'{kind}_width90_a'] = cv, wv, ca, wa
            ks = A.keys_of(d, A.METHODS.get(m, m))
            if ks and ks[0] + '/lam' in d:
                lam = np.concatenate([d[k + '/lam'] for k in ks])
                r['safeguard_lambda_median'], r['safeguard_share_below_one'] = float(np.median(lam)), float(np.mean(lam < 1))
            rec['methods'][m] = {k: (None if (isinstance(x, float) and not np.isfinite(x)) else x) for k, x in r.items()}
        out['systems'][system][sc] = rec
json.dump(out, open(os.path.join(HERE, '..', 'results', 'summary.json'), 'w'), indent=1)
print('summary written:', {s: len(v) for s, v in out['systems'].items()})
