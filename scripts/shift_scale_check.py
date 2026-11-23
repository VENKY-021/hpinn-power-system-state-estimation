"""Median noise-scale estimate of the correction stage (true value 1) with and without the safeguard, nominal and unseen
conditions: shows that a prior-data conflict is read as additional noise.  Output ../results/shift_scale.json.
  python shift_scale_check.py [SYSTEM ...]
"""
import os
import sys
import json
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import evaluate as EV
import estimators as E
import nn as NNm
from data import get_grid

OUT = os.path.join(HERE, '..', 'results', 'shift_scale.json')
N = 100


def main(systems):
    NNm.lower_priority()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for system in systems:
        hp = json.load(open(os.path.join(EV.OUT, f'{system}_hp.json')))
        mu_e, Lam_e = EV.get_prior(system, '')
        g = get_grid(system)
        net, ck = NNm.load_model(system, 'pinn_s0')
        res[system] = {}
        for sc in ('nominal', 'shift'):
            d = NNm.load(system, 'test_' + sc)
            d = {k: v[:N] for k, v in d.items()}
            mu, sd = NNm.predict(net, ck, system, d)
            o = {}
            for name, ev in (('hyep_noev', False), ('hyep', True)):
                s_hat = [E.hpinn_ep(g, d['Z'][k], d['MASK'][k], d['SIG'][k], mu[k], sd[k], mu_e, Lam_e, gamma=hp['gamma2'], kappa=hp['kappa2'], evidence=ev)['scale']
                         for k in range(N)]
                o[name] = float(np.median(s_hat))
            o['eprob'] = float(np.median([E.ep_rwls(g, d['Z'][k], d['MASK'][k], d['SIG'][k], mu_e, Lam_e, kappa=hp['kappa'])['scale'] for k in range(N)]))
            res[system][sc] = o
            print(system, sc, {k: round(v, 3) for k, v in o.items()}, flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:] or ['ieee14', 'ieee30', 'ieee118'])
