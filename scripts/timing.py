"""Run time and iteration counts per snapshot (single thread, same Jacobian code and linear solvers for all estimators)."""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys
import json
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(1)
import nn as NNm
import estimators as E
import evaluate as EV
from data import get_grid

OUT = os.path.join(HERE, '..', 'results', 'timing.json')
N_SNAP = 60
REPEAT = 3


def timed(fn):
    best = 1e9
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        r = fn()
        best = min(best, time.perf_counter() - t0)
    return best, r


def main(systems):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for system in systems:
        g = get_grid(system)
        mu_e, Lam_e = EV.get_prior(system, '')
        hpj = json.load(open(os.path.join(EV.OUT, f'{system}_hp.json')))
        net, ck = NNm.load_model(system, 'pinn_s0')
        d = NNm.load(system, 'test_nominal')
        d = {k: v[:N_SNAP] for k, v in d.items()}
        rows = {k: [] for k in ('wls', 'rwls', 'ep_rwls', 'nn', 'dnn_gn', 'hy', 'hy_nocov', 'hyep_noev', 'safeguard', 'hyep', 'hyep_nocov')}
        its = {k: [] for k in rows}
        for k in range(N_SNAP):
            z, m, s = d['Z'][k], d['MASK'][k], d['SIG'][k]
            one = {kk: v[k:k + 1] for kk, v in d.items() if kk in ('Z', 'MASK', 'SIG')}

            def infer():
                inp = torch.tensor(NNm.make_inputs(g, one, ck['nrm'], ck['cfg']['desc']))
                with torch.no_grad():
                    mu_s, lv = net(inp)
                mu = ck['nrm']['xmu'] + ck['nrm']['xsd'] * mu_s.numpy()[0].astype(np.float64)
                sd = ck['nrm']['xsd'] * np.exp(0.5 * lv.numpy()[0].astype(np.float64))
                return mu, sd

            def fused(p):
                Lnn = E.type_vec(g, *E.as_pair(hpj['gamma2'])) / np.maximum(p[1], 1e-6) ** 2
                Lep = E.scaled_prior(g, Lam_e, hpj['kappa2'])
                Lam = Lep + np.diag(Lnn)
                return np.linalg.solve(Lam, Lnn * p[0] + Lep @ mu_e), Lam

            def run(name, fn):
                t, r = timed(fn)
                rows[name].append(t * 1e3)
                its[name].append(r['iters'] if isinstance(r, dict) else 0)

            run('wls', lambda: E.wls(g, z, m, s))
            run('rwls', lambda: E.rwls(g, z, m, s))
            run('ep_rwls', lambda: E.ep_rwls(g, z, m, s, mu_e, Lam_e, kappa=hpj['kappa']))
            run('nn', lambda: (infer(), dict(iters=0))[1])
            run('dnn_gn', lambda: E.wls(g, z, m, s, x0=infer()[0]))
            run('hy', lambda: (lambda p: E.hpinn(g, z, m, s, p[0], p[1], gamma=hpj['gamma'], want_cov=True))(infer()))
            run('hy_nocov', lambda: (lambda p: E.hpinn(g, z, m, s, p[0], p[1], gamma=hpj['gamma']))(infer()))
            kw = dict(gamma=hpj['gamma2'], kappa=hpj['kappa2'])
            run('hyep_noev', lambda: (lambda p: E.hpinn_ep(g, z, m, s, p[0], p[1], mu_e, Lam_e, want_cov=True, **kw))(infer()))
            pp = infer()                                                    # the safeguard alone: fused prior and evidence search, without the forward pass
            run('safeguard', lambda: dict(iters=0, lam=E.prior_evidence(g, z, m, s, *fused(pp))))
            run('hyep', lambda: (lambda p: E.hpinn_ep(g, z, m, s, p[0], p[1], mu_e, Lam_e, want_cov=True, evidence=True, **kw))(infer()))
            run('hyep_nocov', lambda: (lambda p: E.hpinn_ep(g, z, m, s, p[0], p[1], mu_e, Lam_e, evidence=True, **kw))(infer()))
        # the proposed estimator under a prior-data conflict (unseen operating conditions): the empirical prior is tested as well
        ds = NNm.load(system, 'test_shift')
        ds = {k: v[:N_SNAP] for k, v in ds.items()}
        rows['hyep_shift'], its['hyep_shift'] = [], []
        for k in range(N_SNAP):
            z, m, s = ds['Z'][k], ds['MASK'][k], ds['SIG'][k]
            one_s = {kk: v[k:k + 1] for kk, v in ds.items() if kk in ('Z', 'MASK', 'SIG')}
            inp = torch.tensor(NNm.make_inputs(g, one_s, ck['nrm'], ck['cfg']['desc']))
            with torch.no_grad():
                mu_s, lv = net(inp)
            pm = ck['nrm']['xmu'] + ck['nrm']['xsd'] * mu_s.numpy()[0].astype(np.float64)
            ps = ck['nrm']['xsd'] * np.exp(0.5 * lv.numpy()[0].astype(np.float64))
            t, r = timed(lambda: E.hpinn_ep(g, z, m, s, pm, ps, mu_e, Lam_e, want_cov=True, evidence=True, **kw))
            rows['hyep_shift'].append(t * 1e3)
            its['hyep_shift'].append(r['iters'])
        res[system] = {k: dict(ms_median=float(np.median(v)), ms_mean=float(np.mean(v)), iters=float(np.mean(its[k]))) for k, v in rows.items()}
        print(system, {k: round(v['ms_median'], 1) for k, v in res[system].items()}, flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    NNm.lower_priority()
    main(sys.argv[1:] or ['ieee14', 'ieee30', 'ieee118'])
