"""Does the residual-based noise-scale estimate of the correction stage track the true noise level?
For every noise scenario: mean and median of the estimated scale of the H-PINN (proposed) against the true scale eta."""
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

OUT = os.path.join(HERE, '..', 'results', 'noise_scale.json')
TAGS = ['noise0.5', 'nominal', 'noise2', 'noise3', 'noise4', 'contam', 'bias6', 'param5']
N = 80


def main(systems):
    NNm.lower_priority()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for system in systems:
        g = get_grid(system)
        hpj = json.load(open(os.path.join(EV.OUT, f'{system}_hp.json')))
        mu_e, Lam_e = EV.get_prior(system, '')
        net, ck = NNm.load_model(system, 'pinn_s0')
        res[system] = {}
        for tag in TAGS:
            d = NNm.load(system, 'test_' + tag)
            d = {k: v[:N] for k, v in d.items()}
            mu, sd = NNm.predict(net, ck, system, d)
            sc = []
            for k in range(N):
                r = E.hpinn_ep(g, d['Z'][k], d['MASK'][k], d['SIG'][k], mu[k], sd[k], mu_e, Lam_e, gamma=hpj['gamma2'], kappa=hpj['kappa2'])
                sc.append(r['scale'])
            eta = float(np.mean(d['INFO'][:, 0]))
            res[system][tag] = dict(true_eta=eta, est_mean=float(np.mean(sc)), est_median=float(np.median(sc)), est_p10=float(np.quantile(sc, 0.1)), est_p90=float(np.quantile(sc, 0.9)))
            print(system, tag, {k: round(v, 2) for k, v in res[system][tag].items()}, flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:] or ['ieee14', 'ieee30'])
