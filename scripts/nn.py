"""
Neural estimators (PyTorch, CPU).  One network family for every learning-based method.
  inputs  : standardised measurements (missing -> 0), availability mask, log accuracy-class factor      (3 M)
  outputs : standardised state mean and log-variance                                                    (2 nx)
Variants (flags):
  dnn         supervised Gaussian NLL only (data-driven baseline, no physics)
  pinn        NLL + AC measurement-residual loss (label free) + zero-injection / operating-limit loss, adaptive balancing
  pinn_fixed  as pinn, fixed loss weights                 pinn_noconf  uniform measurement weights (no sigma weighting)
  pinn_nodesc as pinn, no availability / accuracy descriptors in the input
  pinn_nomeas only zero-injection / limit physics         pinn_nophys  measurement loss only
"""
import os
import sys
import json
import time
import ctypes
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import get_grid, DATA
from pypower.idx_brch import F_BUS

RES = os.path.join(HERE, '..', 'results')
MODELS = os.path.join(RES, os.environ.get('NN_MODELS_DIR', 'models'))
MODELS_INIT = os.path.join(RES, 'models')
LR0_ENV = float(os.environ.get('NN_LR', '0'))
BETA = float(os.environ.get('NN_BETA', '0'))              # beta-NLL exponent (0 = plain negative log-likelihood)
torch.set_num_threads(int(os.environ.get('NN_THREADS', '1')))

HP = {'ieee14': dict(hid=128, nl=4, epochs=100, bs=256), 'ieee30': dict(hid=192, nl=4, epochs=100, bs=256),
      'ieee118': dict(hid=256, nl=4, epochs=80, bs=256)}
VARIANTS = {
    'dnn': dict(meas=False, phys=False, adapt=False, conf=True, desc=True),
    'pinn': dict(meas=True, phys=True, adapt=True, conf=True, desc=True),
    'pinn_fixed': dict(meas=True, phys=True, adapt=False, conf=True, desc=True),
    'pinn_noconf': dict(meas=True, phys=True, adapt=True, conf=False, desc=True),
    'pinn_nodesc': dict(meas=True, phys=True, adapt=True, conf=True, desc=False),
    'pinn_nomeas': dict(meas=False, phys=True, adapt=False, conf=True, desc=True),
    'pinn_nophys': dict(meas=True, phys=False, adapt=True, conf=True, desc=True),
}
HUBER_NN = 2.0
SIG_PHYS = 0.02
V_LIM = (0.9, 1.1)
LR0, LR_FACTOR, LR_PATIENCE, LR_MIN = 1e-3, 0.5, 5, 2e-5
PATIENCE, MIN_EPOCHS, WARMUP = 15, 40, 3
W_FIXED = 0.1


def lower_priority():
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)   # below normal
    except Exception:
        pass


def load(system, tag):
    return dict(np.load(os.path.join(DATA, system, f'set_{tag}.npz')))


class Phys:
    """Differentiable AC measurement function h(x); same measurement ordering as grid.Grid.h."""

    def __init__(self, g, dtype=torch.float32):
        Y, Yf = g.Ybus.toarray(), g.Yf.toarray()
        t = lambda a: torch.tensor(a, dtype=dtype)
        self.G, self.B, self.Gf, self.Bf = t(Y.real), t(Y.imag), t(Yf.real), t(Yf.imag)
        self.fb = torch.tensor(g.branch[:, F_BUS].astype(int), dtype=torch.long)
        self.nonref = torch.tensor(g.nonref, dtype=torch.long)
        self.N = g.N
        self.mV, self.mI, self.mF, self.mP, self.zi = [torch.tensor(a, dtype=torch.long) for a in (g.mV, g.mI, g.mF, g.mP, g.zi)]
        self.nzi = len(g.zi)

    def h(self, x):
        B = x.shape[0]
        th = x.new_zeros(B, self.N)
        th[:, self.nonref] = x[:, :self.N - 1]
        Vm = x[:, self.N - 1:]
        e, f = Vm * torch.cos(th), Vm * torch.sin(th)
        Ir, Ii = e @ self.G.T - f @ self.B.T, e @ self.B.T + f @ self.G.T
        P, Q = e * Ir + f * Ii, f * Ir - e * Ii
        Ifr, Ifi = e @ self.Gf.T - f @ self.Bf.T, e @ self.Bf.T + f @ self.Gf.T
        ef, ff = e[:, self.fb], f[:, self.fb]
        Pf, Qf = ef * Ifr + ff * Ifi, ff * Ifr - ef * Ifi
        return torch.cat([Vm[:, self.mV], P[:, self.mI], Q[:, self.mI], Pf[:, self.mF], Qf[:, self.mF],
                          Vm[:, self.mP], th[:, self.mP], P[:, self.zi], Q[:, self.zi]], 1)


class Net(nn.Module):
    def __init__(self, din, nx, hid, nl):
        super().__init__()
        layers, d = [], din
        for _ in range(nl):
            layers += [nn.Linear(d, hid), nn.SiLU()]
            d = hid
        self.trunk = nn.Sequential(*layers)
        self.mu = nn.Linear(hid, nx)
        self.lv = nn.Linear(hid, nx)
        nn.init.zeros_(self.lv.weight)
        nn.init.zeros_(self.lv.bias)

    def forward(self, x):
        h = self.trunk(x)
        return self.mu(h), self.lv(h).clamp(-10.0, 3.0)


def norm_stats(g, d):
    Z, M = d['Z'], d['MASK']
    cnt = M.sum(0).clip(1)
    zmu = (Z * M).sum(0) / cnt
    zsd = np.sqrt((((Z - zmu) ** 2) * M).sum(0) / cnt)
    zsd = np.where(zsd < 1e-6, 1.0, zsd)
    xmu = d['X'].mean(0)
    xsd = np.maximum(d['X'].std(0), 1e-3)
    return dict(zmu=zmu, zsd=zsd, xmu=xmu, xsd=xsd)


def make_inputs(g, d, nrm, desc=True):
    M = d['MASK']
    zt = np.where(M, (d['Z'] - nrm['zmu']) / nrm['zsd'], 0.0)
    if not desc:
        return zt.astype(np.float32)
    lc = np.log(d['SIG'] / g.sigma)
    return np.concatenate([zt, M.astype(float), lc], 1).astype(np.float32)


def train(system, variant, seed, nlab=None, prefix='', log=print, max_epochs=None):
    """Train one model; returns the log dict and saves weights next to it."""
    lower_priority()
    cfg = VARIANTS[variant]
    hp = HP[system]
    g = get_grid(system)
    tr, va = load(system, prefix + 'train'), load(system, prefix + 'val')
    nrm = norm_stats(g, tr)
    K = len(tr['X'])
    nlab = K if nlab is None else int(nlab)
    torch.manual_seed(seed)
    np.random.seed(seed)
    phys = Phys(g)
    net = Net((3 if cfg['desc'] else 1) * g.M, g.nx, hp['hid'], hp['nl'])
    init = os.environ.get('NN_INIT')                      # optional: start from a supervised (pre-trained) network
    if init:
        net.load_state_dict(torch.load(os.path.join(MODELS_INIT, system, init + '.pt'), weights_only=False)['state'])
    s_par = nn.Parameter(torch.zeros(2))
    opt = torch.optim.Adam(list(net.parameters()) + ([s_par] if cfg['adapt'] else []), lr=LR0_ENV or LR0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    tin, vin = T(make_inputs(g, tr, nrm, cfg['desc'])), T(make_inputs(g, va, nrm, cfg['desc']))
    txs, vxs = T((tr['X'] - nrm['xmu']) / nrm['xsd']), T((va['X'] - nrm['xmu']) / nrm['xsd'])
    tz, tm, ts = T(tr['Z']), T(tr['MASK'].astype(float)), T(tr['SIG'])
    xmu, xsd = T(nrm['xmu']), T(nrm['xsd'])
    zi_mask = torch.tensor(g.is_zi)
    nonzi = (~zi_mask).float()
    nzi2 = 2 * phys.nzi
    iV = slice(g.N - 1, g.nx)
    lab = torch.zeros(K)
    lab[:nlab] = 1.0
    # data-driven baseline can only use the labelled samples; physics variants also use the unlabelled measurements
    use = np.arange(K) if (cfg['meas'] or cfg['phys']) else np.arange(nlab)
    bs, nb = hp['bs'], int(np.ceil(len(use) / hp['bs']))
    best, best_state, bad, hist = 1e9, None, 0, []
    t0 = time.time()
    E = max_epochs or hp['epochs']
    for ep in range(E):
        net.train()
        perm = np.random.permutation(use)
        tot = np.zeros(4)
        for b in range(nb):
            idx = torch.tensor(perm[b * bs:(b + 1) * bs])
            mu_s, lv = net(tin[idx])
            lb = lab[idx]
            nll_el = 0.5 * (lv + (txs[idx] - mu_s) ** 2 * torch.exp(-lv))
            if BETA > 0:                                   # beta-NLL: do not let large variances switch off the mean fit
                nll_el = nll_el * torch.exp(lv.detach()) ** BETA
            nll = nll_el.mean(1)
            if ep < WARMUP:                                # warm-up: fit the mean first
                nll = (0.5 * (txs[idx] - mu_s) ** 2).mean(1)
            L_state = (nll * lb).sum() / lb.sum().clamp(min=1.0)
            loss = L_state
            L_meas = L_phys = torch.zeros(())
            if cfg['meas'] or cfg['phys']:
                xh = xmu + xsd * mu_s
                hx = phys.h(xh)
                if cfg['meas']:
                    sg = ts[idx] if cfg['conf'] else torch.full_like(ts[idx], 0.01)
                    u = (tz[idx] - hx) / sg
                    au = u.abs()
                    rho = torch.where(au <= HUBER_NN, 0.5 * u ** 2, HUBER_NN * (au - 0.5 * HUBER_NN))
                    w = tm[idx] * nonzi
                    L_meas = (rho * w).sum() / w.sum().clamp(min=1.0)
                if cfg['phys']:
                    zr = hx[:, -nzi2:] / SIG_PHYS if nzi2 > 0 else hx.new_zeros(1)
                    vb = xh[:, iV]
                    lim = (torch.relu(vb - V_LIM[1]) ** 2 + torch.relu(V_LIM[0] - vb) ** 2) / SIG_PHYS ** 2
                    L_phys = (zr ** 2).mean() + lim.mean()
                if cfg['adapt']:
                    s = s_par.clamp(-5.0, 5.0)
                    if cfg['meas']:
                        loss = loss + torch.exp(-s[0]) * L_meas + s[0]
                    if cfg['phys']:
                        loss = loss + torch.exp(-s[1]) * L_phys + s[1]
                else:
                    loss = loss + (W_FIXED * L_meas if cfg['meas'] else 0.0) + (W_FIXED * L_phys if cfg['phys'] else 0.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += [float(loss.detach()), float(L_state.detach()), float(L_meas.detach()), float(L_phys.detach())]
        net.eval()
        with torch.no_grad():
            mu_v, lv_v = net(vin)
            vnll = float((0.5 * (lv_v + (vxs - mu_v) ** 2 * torch.exp(-lv_v))).mean())
            err = (mu_v - vxs) * xsd
            mae_v = float(err[:, iV].abs().mean())
            mae_a = float(np.rad2deg(err[:, :g.N - 1].abs().mean().item()))
        sched.step(vnll)
        hist.append(dict(epoch=ep + 1, loss=tot[0] / nb, state=tot[1] / nb, meas=tot[2] / nb, phys=tot[3] / nb,
                         val_nll=vnll, val_mae_v=mae_v, val_mae_a=mae_a, lr=opt.param_groups[0]['lr'],
                         time=time.time() - t0))
        if ep >= WARMUP and vnll < best - 1e-4:
            best, bad = vnll, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        elif ep >= WARMUP:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            log(f'{system} {variant} s{seed} ep{ep + 1} loss {tot[0] / nb:.3f} vnll {vnll:.3f} V {mae_v:.5f} ang {mae_a:.4f} '
                f'lr {opt.param_groups[0]["lr"]:.0e} {time.time() - t0:.0f}s')
        if bad >= PATIENCE and ep >= MIN_EPOCHS:
            break
    net.load_state_dict(best_state)
    name = f'{prefix}{variant}' + (f'_L{nlab}' if nlab < K else '') + f'_s{seed}'
    os.makedirs(os.path.join(MODELS, system), exist_ok=True)
    torch.save(dict(state=net.state_dict(), nrm=nrm, cfg=cfg, hp=hp, variant=variant, nlab=nlab), os.path.join(MODELS, system, name + '.pt'))
    info = dict(system=system, variant=variant, seed=seed, nlab=nlab, epochs=len(hist), best_val_nll=best,
                train_time=time.time() - t0, hist=hist, s=[float(v) for v in s_par.detach()])
    json.dump(info, open(os.path.join(MODELS, system, name + '.json'), 'w'))
    return info


def load_model(system, name):
    ck = torch.load(os.path.join(MODELS, system, name + '.pt'), weights_only=False)
    g = get_grid(system)
    net = Net((3 if ck['cfg']['desc'] else 1) * g.M, g.nx, ck['hp']['hid'], ck['hp']['nl'])
    net.load_state_dict(ck['state'])
    net.eval()
    return net, ck


def predict(net, ck, system, d, batch=2048):
    """Physical-unit mean and standard deviation of the state for every sample of dataset d."""
    g = get_grid(system)
    inp = torch.tensor(make_inputs(g, d, ck['nrm'], ck['cfg']['desc']))
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(inp), batch):
            m, l = net(inp[i:i + batch])
            mus.append(m.numpy()); lvs.append(l.numpy())
    mu, lv = np.concatenate(mus).astype(np.float64), np.concatenate(lvs).astype(np.float64)
    xsd = ck['nrm']['xsd']
    return ck['nrm']['xmu'] + xsd * mu, xsd * np.exp(0.5 * lv)


if __name__ == '__main__':
    system, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nlab = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != '-' else None
    prefix = sys.argv[5] if len(sys.argv) > 5 else ''
    ep = int(os.environ.get('NN_MAX_EPOCHS', '0')) or None
    info = train(system, variant, seed, nlab, prefix, max_epochs=ep)
    print('done', system, variant, seed, f"{info['train_time']:.0f}s", 'epochs', info['epochs'], 'best val nll', round(info['best_val_nll'], 4))
