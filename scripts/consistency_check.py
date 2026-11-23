"""Physical consistency of the estimates on the nominal test snapshots: mean zero-injection mismatch (p.u.) and the root-mean-square
normalised measurement residual of the network outputs (DNN, PINN), WLS and the H-PINN (proposed)."""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimators as E
import nn as NNm
import evaluate as EV
from data import get_grid

OUT = os.path.join(HERE, '..', 'results', 'consistency.json')
N = 100


def metrics(g, X, Z, M, S):
    zi, res = [], []
    for k in range(len(X)):
        hx = g.h(X[k])
        zi.append(np.abs(hx[g.is_zi]).mean())
        keep = M[k] & ~g.is_zi
        res.append(np.sqrt(np.mean(((Z[k] - hx)[keep] / S[k][keep]) ** 2)))
    return float(np.mean(zi)), float(np.mean(res))


def main(systems):
    NNm.lower_priority()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for system in systems:
        g = get_grid(system)
        hpj = json.load(open(os.path.join(EV.OUT, f'{system}_hp.json')))
        mu_e, Lam_e = EV.get_prior(system, '')
        d = NNm.load(system, 'test_nominal')
        d = {k: v[:N] for k, v in d.items()}
        o = {}
        for name in ('dnn_s0', 'pinn_s0'):
            net, ck = NNm.load_model(system, name)
            mu, sd = NNm.predict(net, ck, system, d)
            z, r = metrics(g, mu, d['Z'], d['MASK'], d['SIG'])
            o[name.split('_')[0]] = dict(zi_mismatch=z, residual_rms=r)
            if name == 'pinn_s0':
                xs = {'wls': [], 'hyep': []}
                for k in range(N):
                    rw = E.wls(g, d['Z'][k], d['MASK'][k], d['SIG'][k])
                    xs['wls'].append(rw['x'] if rw['converged'] else E.flat_start(g))
                    rh = E.hpinn_ep(g, d['Z'][k], d['MASK'][k], d['SIG'][k], mu[k], sd[k], mu_e, Lam_e, gamma=hpj['gamma2'], kappa=hpj['kappa2'], evidence=True)
                    xs['hyep'].append(rh['x'] if rh['converged'] else E.flat_start(g))
                for m, v in xs.items():
                    z, r = metrics(g, np.array(v), d['Z'], d['MASK'], d['SIG'])
                    o[m] = dict(zi_mismatch=z, residual_rms=r)
        out[system] = o
        print(system, {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in o.items()}, flush=True)
        json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:] or ['ieee14', 'ieee30', 'ieee118'])
