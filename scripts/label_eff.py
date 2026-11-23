"""Label efficiency: accuracy of the data-driven network, the physics-informed network and a Gaussian prior estimated from
the same number of labelled states, as a function of the number of labelled training states.
Networks see the same measurement snapshots (unlabelled for the physics-informed model) and differ only in the labels."""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nn as NNm
import estimators as E
import evaluate as EV
from data import get_grid, DATA
from train_all import LAB

OUT = os.path.join(HERE, '..', 'results', 'label_eff.json')
SC = ['nominal', 'miss30', 'mix']


def main(systems):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for system in systems:
        g = get_grid(system)
        tr = np.load(os.path.join(DATA, system, 'set_train.npz'))
        K = len(tr['X'])
        levels = list(LAB[system]) + [K]
        hpj = json.load(open(os.path.join(EV.OUT, f'{system}_hp.json')))
        kappa = hpj['kappa']
        out = {}
        for sc in SC:
            d = NNm.load(system, 'test_' + sc)
            n = EV.EVAL_N[system]
            d = {k: v[:n] for k, v in d.items()}
            row = {}
            for nl in levels:
                row[str(nl)] = {}
                for variant in ('dnn', 'pinn'):
                    vals = []
                    for s in range(2):
                        name = f'{variant}_s{s}' if nl == K else f'{variant}_L{nl}_s{s}'
                        if not os.path.exists(os.path.join(NNm.MODELS, system, name + '.pt')):
                            continue
                        net, ck = NNm.load_model(system, name)
                        mu, _ = NNm.predict(net, ck, system, d)
                        e = EV.per_sample_errors(g, d['X'], mu)
                        vals.append((e['mae_v'].mean(), e['mae_a'].mean()))
                    if vals:
                        v = np.array(vals)
                        row[str(nl)][variant] = dict(v=float(v[:, 0].mean()), a=float(v[:, 1].mean()), n=len(vals))
                # empirical Gaussian prior (EP-WLS) estimated from the first nl labelled states only
                mu_e, Lam_e = E.empirical_prior(tr['X'][:nl], g)
                errs = []
                for k in range(min(n, 120)):
                    r = E.ep_wls(g, d['Z'][k], d['MASK'][k], d['SIG'][k], mu_e, Lam_e, kappa=kappa)
                    xh = r['x'] if r['converged'] else E.flat_start(g)
                    e = E.errors(g, xh, d['X'][k])
                    errs.append((e['mae_v'], e['mae_a']))
                errs = np.array(errs)
                row[str(nl)]['ep'] = dict(v=float(errs[:, 0].mean()), a=float(errs[:, 1].mean()), n=1)
            out[sc] = row
            print(system, sc, {k: {m: round(v['v'] * 1e3, 3) for m, v in r.items()} for k, r in row.items()}, flush=True)
        res[system] = out
        json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:] or ['ieee14', 'ieee30', 'ieee118'])
