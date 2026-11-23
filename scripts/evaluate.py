"""
Evaluation of all estimators on all test scenarios of one benchmark system.
  python evaluate.py SYSTEM tune        -> choose kappa (empirical prior), gamma (H-PINN prior scale), lambda_c on validation data
  python evaluate.py SYSTEM run [uni]   -> calibration set + all test scenarios; per-sample results in ../results/eval/SYSTEM
Methods: wls, rwls, lnr (subset), ep (empirical-prior WLS), eprob (empirical-prior Huber MAP), per network: nn (output only),
gn (WLS from the network estimate), hy (H-PINN: network prior + robust MAP correction), hyfix (fixed prior precision), plus
raw / conformal intervals.  Every result of a failed run is the flat start (flag stored).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):      # one BLAS thread per worker: no oversubscription
    os.environ[_v] = '1'
import sys
import json
import time
import numpy as np
from multiprocessing import Pool
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import get_grid, scenarios, DATA
import estimators as E
import nn as NNm

OUT = os.path.join(HERE, '..', 'results', 'eval')
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
SEEDS_MAIN = 5
HY_SEEDS = {'ieee14': 3, 'ieee30': 3, 'ieee118': 2}
EVAL_N = {'ieee14': 300, 'ieee30': 300, 'ieee118': 200}
ABL_SCEN = ['nominal', 'miss30', 'bias6', 'shift', 'mix']
ADAPT_SCEN = ['noise3', 'noise4', 'contam']                    # scenarios in which the noise-scale adaptation and the Huber loss matter
LNR_SCEN = ['nominal', 'contam', 'bias3', 'bias6', 'bias10']
ABL_VARIANTS = ['pinn_fixed', 'pinn_noconf', 'pinn_nodesc', 'pinn_nomeas', 'pinn_nophys']
N_WORKERS = int(os.environ.get('EVAL_WORKERS', '4'))

_G = {}


def _init(system, prior):
    NNm.lower_priority()
    _G['g'] = get_grid(system)
    _G['prior'] = prior


def run_sample(args):
    """args: (z, mask, sig, plan, preds) -> dict(name -> (x_hat, iters, ok, std))."""
    z, m, s, plan, preds = args
    g = _G['g']
    mu_e, Lam_e = _G['prior']
    flat = E.flat_start(g)
    out = {}
    for name, spec in plan:
        kind = spec[0]
        if kind == 'wls':
            r = E.wls(g, z, m, s, want_cov=True, cov_scale=True)
        elif kind == 'rwls':
            r = E.rwls(g, z, m, s)
        elif kind == 'lnr':
            r = E.lnr(g, z, m, s)
        elif kind == 'ep':
            r = E.ep_wls(g, z, m, s, mu_e, Lam_e, kappa=spec[1], want_cov=True)
        elif kind == 'eprob':
            r = E.ep_rwls(g, z, m, s, mu_e, Lam_e, kappa=spec[1], want_cov=True)
        elif kind == 'nn':
            mu, sd = preds[spec[1]]
            r = dict(x=mu, iters=0, converged=True, std=sd)
        elif kind == 'gn':
            mu, sd = preds[spec[1]]
            r = E.wls(g, z, m, s, x0=mu)
        elif kind == 'hy':                                   # ('hy', model, gamma, huber, adapt)
            mu, sd = preds[spec[1]]
            r = E.hpinn(g, z, m, s, mu, sd, gamma=spec[2], huber=(E.HUBER if spec[3] else None), adapt=spec[4], want_cov=True)
        elif kind == 'hyep':                                 # ('hyep', model, gamma, kappa, adapt): network prior x empirical prior
            mu, sd = preds[spec[1]]
            r = E.hpinn_ep(g, z, m, s, mu, sd, mu_e, Lam_e, gamma=spec[2], kappa=spec[3], adapt=spec[4], want_cov=True)
        elif kind == 'hyev':                                 # H-PINN (proposed): fused prior with the evidence-based safeguard
            mu, sd = preds[spec[1]]
            r = E.hpinn_ep(g, z, m, s, mu, sd, mu_e, Lam_e, gamma=spec[2], kappa=spec[3], adapt=spec[4], want_cov=True, evidence=True)
        elif kind == 'epev':                                 # EP-RWLS with the same safeguard
            r = E.ep_rwls(g, z, m, s, mu_e, Lam_e, kappa=spec[1], want_cov=True, evidence=True)
        elif kind == 'hyfix':
            mu, sd = preds[spec[1]]
            xsd = spec[3]
            r = E.two_stage(g, z, m, s, mu, prior=(mu, spec[2] / xsd ** 2), want_cov=True)
        else:
            raise ValueError(kind)
        ok = bool(r['converged'])
        out[name] = (r['x'] if ok else flat, r['iters'], ok, r.get('std'), r.get('lam', np.nan), r.get('sel', np.nan))
    return out


def model_names(system, variants, seeds, prefix=''):
    return [f'{prefix}{v}_s{s}' for v in variants for s in range(seeds)]


def load_preds(system, d, names):
    preds = {}
    for n in names:
        path = os.path.join(NNm.MODELS, system, n + '.pt')
        if not os.path.exists(path):
            continue
        net, ck = NNm.load_model(system, n)
        mu, sd = NNm.predict(net, ck, system, d)
        preds[n] = (mu, sd)
    return preds


def per_sample_errors(g, X, XH):
    dv = XH[:, g.N - 1:] - X[:, g.N - 1:]
    da = np.rad2deg(XH[:, :g.N - 1] - X[:, :g.N - 1])
    return dict(mae_v=np.abs(dv).mean(1), mae_a=np.abs(da).mean(1), rmse_v=np.sqrt((dv ** 2).mean(1)), rmse_a=np.sqrt((da ** 2).mean(1)))


def run_plan(system, d, plan, preds, prior, pool, with_x=True):
    """Execute plan on every sample of dataset d; returns dict name -> dict(x, iters, ok, std)."""
    K = len(d['X'])
    tasks = [(d['Z'][k], d['MASK'][k], d['SIG'][k], plan, {n: (p[0][k], p[1][k]) for n, p in preds.items()}) for k in range(K)]
    res = pool.map(run_sample, tasks, chunksize=max(1, K // (N_WORKERS * 6)))
    out = {}
    for name, _ in plan:
        xs = np.array([r[name][0] for r in res])
        it = np.array([r[name][1] for r in res])
        ok = np.array([r[name][2] for r in res])
        sd = [r[name][3] for r in res]
        has = np.array([s is not None and np.all(np.isfinite(s)) for s in sd])
        std = np.array([s if h else np.full(xs.shape[1], np.nan) for s, h in zip(sd, has)])
        lam = np.array([r[name][4] for r in res], dtype=float)
        sel = np.array([r[name][5] for r in res], dtype=float)
        out[name] = dict(x=xs, iters=it, ok=ok, std=std, has_std=has, lam=lam, sel=sel)
    return out


# ------------------------------------------------------------------------------------------------ conformal calibration
def class_index(g, obs):
    """Mondrian classes: (observable, state type) -> index obs*2 + (0 voltage, 1 angle); returns per-sample base obs vector."""
    return obs.astype(int)


def score_matrix(g, X, XH, STD):
    return np.abs(X - XH) / np.maximum(STD, 1e-12)


def calibrate(g, X, XH, STD, obs):
    """Quantiles of the pooled normalised scores per class (observable flag x voltage/angle) for every level."""
    S = score_matrix(g, X, XH, STD)
    th = {}
    for o in (0, 1):
        for t, sl in (('v', slice(g.N - 1, g.nx)), ('a', slice(0, g.N - 1))):
            v = S[obs == o][:, sl].ravel()
            v = v[np.isfinite(v)]
            n = len(v)
            q = []
            for lv in LEVELS:
                if n == 0:
                    q.append(np.nan)
                else:
                    q.append(float(np.quantile(v, min(1.0, np.ceil((n + 1) * lv) / n))))
            th[f'{o}{t}'] = q
    return th


def coverage(g, X, XH, STD, obs, th=None):
    """Per-sample coverage (voltage, angle) at every level and mean interval width; raw Gaussian (th=None) or conformal."""
    A = np.abs(X - XH)
    res = {}
    for t, sl in (('v', slice(g.N - 1, g.nx)), ('a', slice(0, g.N - 1))):
        cov = np.full((len(X), len(LEVELS)), np.nan)
        wid = np.full((len(X), len(LEVELS)), np.nan)
        for j, lv in enumerate(LEVELS):
            if th is None:
                q = np.full(len(X), norm.ppf((1 + lv) / 2))
            else:
                q = np.array([th[f'{int(o)}{t}'][j] for o in obs])
            lim = q[:, None] * STD[:, sl]
            bad = ~np.isfinite(STD[:, sl]).all(1)
            cov[:, j] = np.where(bad, np.nan, (A[:, sl] <= lim).mean(1))
            wid[:, j] = np.where(bad, np.nan, (2 * lim).mean(1))
        res['cov_' + t], res['wid_' + t] = cov, wid
    return res


# ------------------------------------------------------------------------------------------------ plans
def classical_plan(hp, scen, system=''):
    p = [('wls', ('wls',)), ('rwls', ('rwls',)), ('ep', ('ep', hp['kappa'])), ('eprob', ('eprob', hp['kappa'])), ('epev', ('epev', hp['kappa']))]
    if scen in LNR_SCEN and (system != 'ieee118' or scen == 'bias6'):      # LNR is expensive for the 118-bus system
        p.append(('lnr', ('lnr',)))
    return p


def nn_plan(system, hp, scen, names, prefix='', cal=False):
    p = []
    hs = HY_SEEDS[system]
    for n in names:
        v, sd = n.rsplit('_s', 1)
        sd = int(sd)
        v = v[len(prefix):] if prefix and v.startswith(prefix) else v
        base = v.split('_L')[0]
        p.append(('nn:' + n, ('nn', n)))
        if '_L' in v:
            continue
        light = system == 'ieee118'                          # thinned plan for the largest system
        if base == 'pinn' and sd < hs:
            if not light or scen in ABL_SCEN or scen in ADAPT_SCEN or cal:
                p.append(('hy:' + n, ('hy', n, hp['gamma'], True, True)))
            p.append(('hyep:' + n, ('hyep', n, hp['gamma2'], hp['kappa2'], True)))
            p.append(('hyev:' + n, ('hyev', n, hp['gamma2'], hp['kappa2'], True)))
            if scen in ABL_SCEN and not cal:
                p.append(('gn:' + n, ('gn', n)))
        if base == 'dnn' and sd < hs and not cal:
            p.append(('gn:' + n, ('gn', n)))
            if scen in ABL_SCEN:
                p.append(('hy:' + n, ('hy', n, hp['gamma'], True, True)))
        if base in ABL_VARIANTS and sd < (1 if light else 2) and scen in ABL_SCEN and not cal:
            p.append(('hy:' + n, ('hy', n, hp['gamma'], True, True)))
        if base == 'pinn' and sd == 0 and scen in ADAPT_SCEN and not cal:
            p.append(('hyq:' + n, ('hy', n, hp['gamma'], False, True)))
            p.append(('hyna:' + n, ('hy', n, hp['gamma'], True, False)))
        if base == 'pinn' and sd == 0 and scen in ABL_SCEN and not cal:
            p.append(('hyq:' + n, ('hy', n, hp['gamma'], False, True)))            # quadratic correction (no Huber)
            p.append(('hyna:' + n, ('hy', n, hp['gamma'], True, False)))           # no noise-scale adaptation
            p.append(('hyfix:' + n, ('hyfix', n, hp['lamc'], None)))               # fixed prior precision (set below)
    return p


def fix_plan(plan, xsd):
    return [(n, s if s[0] != 'hyfix' else ('hyfix', s[1], s[2], xsd)) for n, s in plan]


# ------------------------------------------------------------------------------------------------ main drivers
def get_prior(system, prefix):
    g = get_grid(system)
    tr = np.load(os.path.join(DATA, system, f'set_{prefix}train.npz'))
    return E.empirical_prior(tr['X'], g)


def tune(system, prefix=''):
    g = get_grid(system)
    prior = get_prior(system, prefix)
    va = NNm.load(system, prefix + 'val')
    n = int(os.environ.get('TUNE_N', '200'))
    d = {k: v[:n] for k, v in va.items()}
    names = [f'{prefix}pinn_s0']
    preds = load_preds(system, d, names)
    xsd = NNm.load_model(system, names[0])[1]['nrm']['xsd']
    score = lambda x: float(np.mean(per_sample_errors(g, d['X'], x)['mae_v']) / 1e-3 + np.mean(per_sample_errors(g, d['X'], x)['mae_a']) / 0.1)
    if os.environ.get('TUNE_STAGE') == 'C':                        # extended search of the fused-prior scales only
        hpj = json.load(open(os.path.join(OUT, f'{prefix}{system}_hp.json')))
        kstar = hpj['kappa']
        plan2 = [(f'hyep|{ga:g}|{gv:g}|1', ('hyep', names[0], (ga, gv), (kstar[0], kstar[1]), True))
                 for ga in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0) for gv in (0.03, 0.1, 0.3, 1.0, 3.0)]
        with Pool(N_WORKERS, initializer=_init, initargs=(system, prior)) as pool:
            res2 = run_plan(system, d, plan2, preds, prior, pool)
        for nme, r in res2.items():
            hpj['table'][nme] = score(r['x'])
            e = per_sample_errors(g, d['X'], r['x'])
            print(f'  {nme:18s} score {hpj["table"][nme]:.3f}  MAE_V {e["mae_v"].mean():.5f}  MAE_ang {e["mae_a"].mean():.4f}', flush=True)
        best = min([k for k in hpj['table'] if k.startswith('hyep|')], key=lambda k: hpj['table'][k])
        _, ga, gv, c = best.split('|')
        hpj['gamma2'], hpj['kappa2'] = [float(ga), float(gv)], [kstar[0] * float(c), kstar[1] * float(c)]
        json.dump(hpj, open(os.path.join(OUT, f'{prefix}{system}_hp.json'), 'w'), indent=1)
        print('chosen fused prior', best, hpj['gamma2'], hpj['kappa2'])
        return
    meta, plan = {}, []
    for ka in (1.0, 3.0, 10.0):                                    # empirical-prior scales: angles, magnitudes
        for kv in (1.0, 3.0, 10.0, 30.0):
            nme = f'eprob|{ka:g}|{kv:g}'
            plan.append((nme, ('eprob', (ka, kv))))
            meta[nme] = ('kappa', [ka, kv])
    for ga in (0.03, 0.1, 0.3, 1.0):                               # network-prior scales: angles, magnitudes
        for gv in (0.3, 1.0, 3.0):
            nme = f'hy|{ga:g}|{gv:g}'
            plan.append((nme, ('hy', names[0], (ga, gv), True, True)))
            meta[nme] = ('gamma', [ga, gv])
    for lc in (100.0, 1000.0, 10000.0):
        nme = f'hyfix|{lc:g}'
        plan.append((nme, ('hyfix', names[0], lc, xsd)))
        meta[nme] = ('lamc', lc)
    with Pool(N_WORKERS, initializer=_init, initargs=(system, prior)) as pool:
        res = run_plan(system, d, plan, preds, prior, pool)
        table = {nme: score(r['x']) for nme, r in res.items()}
        kstar = meta[min([k for k in table if k.startswith('eprob|')], key=lambda k: table[k])][1]
        plan2, meta2 = [], {}
        for ga in (0.03, 0.1, 0.3, 1.0):                              # fused prior: network scales and a multiplier of kappa*
            for gv in (0.3, 1.0, 3.0):
                for c in (0.3, 1.0):
                    nme = f'hyep|{ga:g}|{gv:g}|{c:g}'
                    plan2.append((nme, ('hyep', names[0], (ga, gv), (kstar[0] * c, kstar[1] * c), True)))
                    meta2[nme] = ([ga, gv], [kstar[0] * c, kstar[1] * c])
        res2 = run_plan(system, d, plan2, preds, prior, pool)
    res.update(res2)
    table = {nme: score(r['x']) for nme, r in res.items()}
    for nme, r in res.items():
        e = per_sample_errors(g, d['X'], r['x'])
        print(f'  {nme:14s} score {table[nme]:.3f}  MAE_V {e["mae_v"].mean():.5f}  MAE_ang {e["mae_a"].mean():.4f}  fail {1 - r["ok"].mean():.3f}  it {r["iters"].mean():.1f}')
    hp = {}
    for key, pre in (('kappa', 'eprob|'), ('gamma', 'hy|'), ('lamc', 'hyfix|')):
        b = min([k for k in table if k.startswith(pre)], key=lambda k: table[k])
        hp[key] = meta[b][1]
    b2 = min([k for k in table if k.startswith('hyep|')], key=lambda k: table[k])
    hp['gamma2'], hp['kappa2'] = meta2[b2][0], meta2[b2][1]
    hp['table'] = table
    os.makedirs(OUT, exist_ok=True)
    json.dump(hp, open(os.path.join(OUT, f'{prefix}{system}_hp.json'), 'w'), indent=1)
    print('chosen', {k: v for k, v in hp.items() if k != 'table'})


def run(system, prefix=''):
    g = get_grid(system)
    hp = json.load(open(os.path.join(OUT, f'{prefix}{system}_hp.json')))
    prior = get_prior(system, prefix)
    odir = os.path.join(OUT, prefix + system)
    os.makedirs(odir, exist_ok=True)
    seeds = SEEDS_MAIN
    names = model_names(system, ['pinn', 'dnn'], seeds, prefix) + model_names(system, ABL_VARIANTS, 2, prefix)
    names = [n for n in names if os.path.exists(os.path.join(NNm.MODELS, system, n + '.pt'))]
    xsd = NNm.load_model(system, names[0])[1]['nrm']['xsd']
    t0 = time.time()
    with Pool(N_WORKERS, initializer=_init, initargs=(system, prior)) as pool:
        # calibration set (mixture): scores of every interval-producing method
        skip = os.environ.get('SKIP_EXISTING') == '1'
        thr_path = os.path.join(odir, '_thresholds.json')
        dn = [n for n in names if n.startswith(f'{prefix}dnn_s')]
        ens = lambda dd, pr: (np.mean([pr[n][0] for n in dn], 0), np.std([pr[n][0] for n in dn], 0) + 1e-9)
        only_m = os.environ.get('ONLY_METHODS')                  # re-evaluate only the plan entries with these name prefixes and merge
        sel = tuple(only_m.split(',')) if only_m else None
        if skip and os.path.exists(thr_path) and not sel:
            thr = json.load(open(thr_path))
            print(system, 'calibration thresholds reused', flush=True)
        else:
            cal = NNm.load(system, prefix + 'cal')
            preds = load_preds(system, cal, names)
            plan = fix_plan([p for p in classical_plan(hp, 'mix', system) if p[0] != 'rwls'] + nn_plan(system, hp, 'mix', [n for n in names if n[len(prefix):].split('_s')[0] in ('pinn', 'dnn')], prefix, cal=True), xsd)
            if sel:
                plan = [(n, s) for n, s in plan if n.startswith(sel)]
            rc = run_plan(system, cal, plan, preds, prior, pool)
            new = {nme: calibrate(g, cal['X'], r['x'], r['std'], cal['OBS']) for nme, r in rc.items() if r['has_std'].any()}
            if not sel or 'ens' in sel:
                em, es = ens(cal, preds)                   # ensemble of the data-driven networks
                new['ens'] = calibrate(g, cal['X'], em, es, cal['OBS'])
            thr = json.load(open(thr_path)) if (sel and os.path.exists(thr_path)) else {}
            thr.update(new)
            json.dump(thr, open(thr_path, 'w'))
            print(system, 'calibration done', f'{time.time() - t0:.0f}s', flush=True)
        only = os.environ.get('EVAL_SCEN')
        for scen in scenarios():
            if prefix and scen.startswith('shift'):
                continue
            if only and scen not in only.split(','):
                continue
            if skip and os.path.exists(os.path.join(odir, f'{scen}.npz')):
                continue
            tag = prefix + 'test_' + scen
            d = NNm.load(system, tag)
            n_eval = int(os.environ.get('EVAL_N', EVAL_N[system]))
            d = {k: v[:n_eval] for k, v in d.items()}
            preds = load_preds(system, d, names)
            plan = fix_plan(classical_plan(hp, scen, system) + nn_plan(system, hp, scen, names, prefix), xsd)
            if sel:
                plan = [(n, s) for n, s in plan if n.startswith(sel)]
            r = run_plan(system, d, plan, preds, prior, pool)
            old_f = os.path.join(odir, f'{scen}.npz')
            save = dict(np.load(old_f)) if (sel and os.path.exists(old_f)) else dict(OBS=d['OBS'], META=d['META'], INFO=d['INFO'])
            for nme, rr in r.items():
                kk = nme.replace(':' + prefix, ':') if prefix else nme          # result keys do not carry the data-set prefix
                e = per_sample_errors(g, d['X'], rr['x'])
                for k, v in e.items():
                    save[f'{kk}/{k}'] = v
                save[f'{kk}/iters'], save[f'{kk}/ok'] = rr['iters'], rr['ok']
                if np.isfinite(rr['lam']).any():
                    save[f'{kk}/lam'] = rr['lam']
                if np.isfinite(rr['sel']).any():
                    save[f'{kk}/sel'] = rr['sel']
                if rr['has_std'].any():
                    for tname, th in (('raw', None), ('cp', thr.get(nme))):
                        if tname == 'cp' and th is None:
                            continue
                        c = coverage(g, d['X'], rr['x'], rr['std'], d['OBS'], th)
                        for k, v in c.items():
                            save[f'{kk}/{tname}_{k}'] = v
                        save[f'{kk}/has_std'] = rr['has_std']
            if not sel or 'ens' in sel:
                ee = ens(d, preds)
                e = per_sample_errors(g, d['X'], ee[0])
                for k, v in e.items():
                    save[f'ens/{k}'] = v
                for tname, th in (('raw', None), ('cp', thr['ens'])):
                    c = coverage(g, d['X'], ee[0], ee[1], d['OBS'], th)
                    for k, v in c.items():
                        save[f'ens/{tname}_{k}'] = v
            tmp_f = os.path.join(odir, f'{scen}.tmp.npz')              # write atomically, so that a kill never leaves a truncated file
            np.savez_compressed(tmp_f, **save)
            os.replace(tmp_f, os.path.join(odir, f'{scen}.npz'))
            print(' ', scen, f'{time.time() - t0:.0f}s', flush=True)
    print(system, 'evaluation done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    system, mode = sys.argv[1], sys.argv[2]
    prefix = 'u_' if len(sys.argv) > 3 and sys.argv[3] == 'uni' else ''
    (tune if mode == 'tune' else run)(system, prefix) if mode in ('tune', 'run') else None
