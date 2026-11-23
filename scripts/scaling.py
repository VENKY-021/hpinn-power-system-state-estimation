"""Scaling of the per-iteration cost of the dense Gauss-Newton step and of the network forward pass with the system size
(14, 30, 118 and 300 buses). Single thread; run on an otherwise idle machine."""
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
import estimators as E
import nn as NNm
from grid import Grid

OUT = os.path.join(HERE, '..', 'results', 'scaling.json')
WIDTH = {'ieee14': 128, 'ieee30': 192, 'ieee118': 256, 'ieee300': 256}


def best(fn, rep=5):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def main():
    NNm.lower_priority()
    res = {}
    for name in ('ieee14', 'ieee30', 'ieee118', 'ieee300'):
        g = Grid(name)
        apv = 0.4 if g.N < 200 else 0.0
        Va, Vm, ok, _ = g.solve_pf(1.0, np.ones(g.N), apv, apv)
        assert ok, name
        x = g.x_from(Va, Vm)
        zt = g.true_meas(Va, Vm)
        rng = np.random.default_rng(1)
        z = zt + g.sigma * rng.standard_normal(g.M)
        z[g.is_zi] = zt[g.is_zi]
        mask = np.ones(g.M, bool)
        xs = E.flat_start(g)
        rows = np.arange(g.M)
        w = 1.0 / g.sigma ** 2
        t_h = best(lambda: g.h(xs))
        t_H = best(lambda: g.H(xs))
        Hm = g.H(xs)
        t_G = best(lambda: Hm.T @ (w[:, None] * Hm))
        G = Hm.T @ (w[:, None] * Hm) + np.eye(g.nx) * 1e-6
        b = Hm.T @ (w * (z - g.h(xs)))
        t_chol = best(lambda: E._step(G, b))
        t_inv = best(lambda: np.linalg.inv(G))
        Lam = np.eye(g.nx) * 4.0 + 0.1 * np.ones((g.nx, g.nx)) / g.nx          # dense well-conditioned precision, only to time the safeguard
        t_ev = best(lambda: E.prior_evidence(g, z, mask, g.sigma, xs, Lam), rep=2)
        net = NNm.Net(3 * g.M, g.nx, WIDTH[name], 4)
        inp = torch.zeros(1, 3 * g.M)
        with torch.no_grad():
            t_nn = best(lambda: net(inp))
        t_wls = best(lambda: E.wls(g, z, mask, g.sigma), rep=2)
        r = E.wls(g, z, mask, g.sigma)
        res[name] = dict(N=g.N, nx=g.nx, M=g.M, h_ms=t_h, jac_ms=t_H, gain_ms=t_G, chol_ms=t_chol, inv_ms=t_inv, ev_ms=t_ev, nn_ms=t_nn,
                         wls_ms=t_wls, wls_iters=r['iters'], iter_ms=t_h + t_H + t_G + t_chol, params=sum(p.numel() for p in net.parameters()))
        print(name, {k: round(v, 2) if isinstance(v, float) else v for k, v in res[name].items()}, flush=True)
    names = [k for k in res if k.startswith('ieee')]
    N = np.array([res[k]['N'] for k in names], float)
    for key in ('iter_ms', 'nn_ms', 'chol_ms', 'gain_ms', 'jac_ms', 'ev_ms'):
        y = np.array([res[k][key] for k in names])
        res.setdefault('slope', {})[key] = float(np.polyfit(np.log(N), np.log(y), 1)[0])
    json.dump(res, open(OUT, 'w'), indent=1)
    print('slopes', res['slope'])


if __name__ == '__main__':
    main()
