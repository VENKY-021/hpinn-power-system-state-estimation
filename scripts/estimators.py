"""
Model-based estimators (numpy): Gauss-Newton / IRLS solver for
    J(x) = sum_i m_i rho(r_i / sig_i) + (x - mu)^T Lam (x - mu),     r = z - h(x),
with optional Huber loss (zero-injection rows stay quadratic) and optional Gaussian prior (mu, Lam).
  WLS        : no prior, quadratic loss, flat start
  RWLS       : no prior, two-stage Huber loss, flat start
  EP-WLS     : empirical Gaussian prior of the training states (full covariance), start at the prior mean
  EP-RWLS    : the same prior with the two-stage Huber loss
  NN-GN      : WLS started at the neural-network estimate (learned initialisation only)
  H-PINN     : two-stage Huber MAP with the network's heteroscedastic prior precision, started at the network mean
  LNR        : WLS with largest-normalised-residual bad-data removal
Robust two-stage runs estimate the actual noise scale from the residuals of the first (quadratic) stage with a robust MAD
estimate corrected for the degrees of freedom used by the fit, and inflate the declared standard deviations accordingly.
Failed runs (no convergence, divergence) are flagged; the reported state of a failed run is the flat start.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

HUBER = 1.5
S_MIN, S_MAX = 0.5, 8.0


def flat_start(g):
    x = np.ones(g.nx)
    x[:g.N - 1] = 0.0
    return x


def type_vec(g, a, v):
    """Per-state vector with value a for angles and v for voltage magnitudes."""
    return np.concatenate([np.full(g.N - 1, float(a)), np.full(g.N, float(v))])


def as_pair(p):
    return (p, p) if np.isscalar(p) else (p[0], p[1])


def _step(G, b):
    """Solve G dx = b: Cholesky when well conditioned, eigen-decomposition pseudo-inverse when (nearly) singular."""
    try:
        c = cho_factor(G, lower=True, check_finite=False)
        d = np.abs(np.diag(c[0]))
        if d.min() / d.max() > 1e-7:
            return cho_solve(c, b, check_finite=False), True
    except np.linalg.LinAlgError:
        pass
    w, V = eigh(G)
    keep = w > 1e-10 * w.max()
    return V[:, keep] @ ((V[:, keep].T @ b) / w[keep]), False


def noise_scale(g, z, mask, sig, x, prior=None):
    """Robust estimate of the ratio between actual and declared noise: MAD of the standardised residuals, corrected for the
    degrees of freedom absorbed by the fit (effective number of parameters tr[(G + Lam)^-1 G])."""
    rows = np.where(mask)[0]
    nz = ~g.is_zi[rows]
    Ma = int(nz.sum())
    if Ma < 5:
        return 1.0
    u = (z[rows] - g.h(x)[rows]) / sig[rows]
    Hm = g.H(x)[rows]
    w = 1.0 / sig[rows] ** 2
    Gd = Hm.T @ (w[:, None] * Hm)
    Gp = Gd.copy()
    if prior is not None:
        Lam = prior[1]
        if Lam.ndim == 1:
            Gp[np.diag_indices_from(Gp)] += Lam
        else:
            Gp += Lam
    try:
        peff = float(np.trace(np.linalg.solve(Gp, Gd)))
    except np.linalg.LinAlgError:
        return 1.0
    nzr = int((~nz).sum())
    dof = max(Ma - (peff - nzr), 0.25 * Ma)
    s = 1.4826 * float(np.median(np.abs(u[nz]))) / np.sqrt(dof / Ma)
    return float(np.clip(s, S_MIN, S_MAX))


EV_LAMS = 2.0 ** -np.arange(0, 17)                                   # prior-precision multipliers searched by the safeguard (never above 1)
EV_SCALES = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0])  # noise scales searched jointly (nuisance parameter)


def evidence_max(g, z, mask, sig, mu, Lam):
    """Evidence-based prior-conflict safeguard.  The innovation r0 = z - h(mu) of the prior mean has, under the linearised
    model, the covariance S = H Lam^-1 H^T / lam + s^2 R, where lam scales the prior precision and s the declared noise.
    The multiplier lam in {1, 1/2, ..., 2^-16} that maximises the marginal likelihood -(r0^T S^-1 r0 + log det S) / 2 is
    returned (s is searched jointly and discarded).  With the eigen-decomposition of the whitened H Lam^-1 H^T = Q diag(rho) Q^T
    the likelihood of every (lam, s) costs O(M).  lam = 1 means that prior and data agree; a small lam means a prior-data conflict."""
    rows = np.where(mask)[0]
    if len(rows) < 5:
        return -np.inf, 1.0
    try:
        Hm = g.H(mu)[rows]
        sg = sig[rows]
        A = Hm @ np.linalg.solve(Lam, Hm.T) / np.outer(sg, sg)
        rho, Q = np.linalg.eigh(0.5 * (A + A.T))
    except np.linalg.LinAlgError:
        return -np.inf, 1.0
    rho = np.clip(rho, 0.0, None)
    c = Q.T @ ((z[rows] - g.h(mu)[rows]) / sg)
    d = rho[None, None, :] / EV_LAMS[None, :, None] + (EV_SCALES ** 2)[:, None, None]
    ll = -0.5 * np.sum(c[None, None, :] ** 2 / d + np.log(d), axis=2)
    k = np.unravel_index(np.argmax(ll), ll.shape)
    return float(ll[k]), float(EV_LAMS[k[1]])


def prior_evidence(g, z, mask, sig, mu, Lam):
    """Evidence-based precision multiplier lam of a single prior (the safeguard of EP-RWLS)."""
    return evidence_max(g, z, mask, sig, mu, Lam)[1]


def solve(g, z, mask, sig, x0, huber=None, prior=None, maxit=40, tol=1e-5, want_cov=False, cov_scale=False):
    rows = np.where(mask)[0]
    zi = g.is_zi[rows]
    zr, sr = z[rows], sig[rows]
    w0 = 1.0 / sr ** 2
    x = x0.copy()
    if prior is not None:
        mu, Lam = prior
        diag = Lam.ndim == 1

    def cost(x, hx):
        u = np.abs(zr - hx[rows]) / sr
        if huber is None:
            c = 0.5 * np.sum(u ** 2)
        else:
            c = np.sum(np.where(zi | (u <= huber), 0.5 * u ** 2, huber * (u - 0.5 * huber)))
        if prior is not None:
            d = x - mu
            c += 0.5 * (np.sum(Lam * d * d) if diag else d @ Lam @ d)
        return c

    hx = g.h(x)
    J = cost(x, hx)
    ok_chol, it, conv = True, 0, False
    G = None
    for it in range(1, maxit + 1):
        r = zr - hx[rows]
        w = w0
        if huber is not None:
            u = np.abs(r) / sr
            w = np.where(zi | (u <= huber), w0, w0 * huber / np.maximum(u, 1e-12))
        Hm = g.H(x)[rows]
        G = Hm.T @ (w[:, None] * Hm)
        b = Hm.T @ (w * r)
        if prior is not None:
            if diag:
                G[np.diag_indices_from(G)] += Lam
                b -= Lam * (x - mu)
            else:
                G += Lam
                b -= Lam @ (x - mu)
        dx, ok_chol = _step(G, b)
        t, accepted = 1.0, False
        for _ in range(8):
            xn = x + t * dx
            hn = g.h(xn)
            Jn = cost(xn, hn)
            if np.isfinite(Jn) and Jn <= J * (1 + 1e-12) + 1e-12:
                accepted = True
                break
            t *= 0.5
        if not accepted:                      # no descent possible: numerical stationary point
            conv = True
            break
        dJ, step = J - Jn, float(np.max(np.abs(t * dx)))
        x, hx, J = xn, hn, Jn
        if step < tol or dJ <= 1e-9 * max(1.0, abs(J)):
            conv = True
            break
        if not np.all(np.isfinite(x)) or np.max(np.abs(x[g.N - 1:] - 1.0)) > 1.0:
            break
    ok = bool(conv and np.all(np.isfinite(x)) and np.max(np.abs(x[g.N - 1:] - 1.0)) < 0.6)
    out = dict(x=x, iters=it, converged=ok, J=float(J), chol=bool(ok_chol), scale=1.0)
    if want_cov and G is not None and (prior is not None or ok_chol):
        try:
            std = np.sqrt(np.clip(np.diag(np.linalg.inv(G)), 0, None))
            if cov_scale:
                s = noise_scale(g, z, mask, sig, x, prior)
                std = std * s
                out['scale'] = s
            out['std'] = std
        except np.linalg.LinAlgError:
            out['std'] = np.full(g.nx, np.nan)
    return out


def wls(g, z, mask, sig, x0=None, **kw):
    return solve(g, z, mask, sig, flat_start(g) if x0 is None else x0, **kw)


def two_stage(g, z, mask, sig, x0, prior=None, huber=HUBER, adapt=True, **kw):
    """Quadratic Gauss-Newton stage (reaches the basin and gives the residuals for the noise-scale estimate), then the
    robust stage started from that solution with the declared standard deviations inflated by the estimated noise scale."""
    r1 = solve(g, z, mask, sig, x0, huber=None, prior=prior)
    s = noise_scale(g, z, mask, sig, r1['x'], prior) if (adapt and np.all(np.isfinite(r1['x']))) else 1.0
    r2 = solve(g, z, mask, sig * s, r1['x'], huber=huber, prior=prior, **kw)
    r2['iters'] += r1['iters']
    r2['converged'] = bool(r2['converged'] and np.all(np.isfinite(r1['x'])))
    r2['scale'] = s
    return r2


def rwls(g, z, mask, sig, x0=None, **kw):
    return two_stage(g, z, mask, sig, flat_start(g) if x0 is None else x0, **kw)


def scaled_prior(g, Lam_full, kappa):
    """Empirical-prior precision with separate scales for angles and magnitudes (symmetric scaling keeps it PSD)."""
    ka, kv = as_pair(kappa)
    d = np.sqrt(type_vec(g, ka, kv))
    return Lam_full * d[:, None] * d[None, :]


def ep_wls(g, z, mask, sig, mu, Lam, kappa=1.0, **kw):
    return solve(g, z, mask, sig, mu, prior=(mu, scaled_prior(g, Lam, kappa)), cov_scale=True, **kw)


def ep_rwls(g, z, mask, sig, mu, Lam, kappa=1.0, evidence=False, **kw):
    Lp = scaled_prior(g, Lam, kappa)
    lam = prior_evidence(g, z, mask, sig, mu, Lp) if evidence else 1.0
    r = two_stage(g, z, mask, sig, mu, prior=(mu, lam * Lp), **kw)
    r['lam'] = lam
    return r


def empirical_prior(X_train, g, jitter=1e-3, floor_v=1e-3, floor_a=1e-3):
    """Gaussian prior from the training states: mean, full covariance, relative jitter and absolute variance floors."""
    mu = X_train.mean(0)
    S = np.cov(X_train.T)
    fl = np.concatenate([np.full(g.N - 1, floor_a ** 2), np.full(g.N, floor_v ** 2)])
    S = S + jitter * np.diag(np.diag(S)) + np.diag(fl)
    return mu, np.linalg.inv(S)


def hpinn(g, z, mask, sig, mu_nn, std_nn, gamma=1.0, huber=HUBER, adapt=True, **kw):
    ga, gv = as_pair(gamma)
    Lam = type_vec(g, ga, gv) / np.maximum(std_nn, 1e-6) ** 2
    return two_stage(g, z, mask, sig, mu_nn, prior=(mu_nn, Lam), huber=huber, adapt=adapt, **kw)


def hpinn_ep(g, z, mask, sig, mu_nn, std_nn, mu_ep, Lam_ep, gamma=1.0, kappa=1.0, huber=HUBER, adapt=True, evidence=False, **kw):
    """H-PINN with the product of two Gaussian priors: the network prior (diagonal precision gamma / std^2) and the empirical
    prior of the training states (dense precision kappa * Lam_ep); the fused prior mean is the precision-weighted mean.
    With evidence=True the prior precision is multiplied by the evidence-based safeguard factor (prior_evidence)."""
    ga, gv = as_pair(gamma)
    Lnn = type_vec(g, ga, gv) / np.maximum(std_nn, 1e-6) ** 2
    Lep = scaled_prior(g, Lam_ep, kappa)
    Lam = Lep + np.diag(Lnn)
    mu = np.linalg.solve(Lam, Lnn * mu_nn + Lep @ mu_ep)
    lam, sel = 1.0, 1.0
    if evidence:                       # evidence-based choice between the fused and the empirical prior, and of the precision multiplier
        ll_f, lam_f = evidence_max(g, z, mask, sig, mu, Lam)
        lam = lam_f
        if lam_f < 1.0:                # conflict: the empirical prior alone is tested as well, the one with the larger evidence is used
            ll_e, lam_e = evidence_max(g, z, mask, sig, mu_ep, Lep)
            if ll_e > ll_f:
                mu, Lam, lam, sel = mu_ep, Lep, lam_e, 0.0
    r = two_stage(g, z, mask, sig, mu, prior=(mu, lam * Lam), huber=huber, adapt=adapt, **kw)
    r['lam'], r['sel'] = lam, sel
    return r


def lnr(g, z, mask, sig, thr=3.0, max_rm=10):
    """WLS with largest-normalised-residual test; flagged measurements are removed one at a time."""
    mask = mask.copy()
    removed = 0
    res = wls(g, z, mask, sig)
    for _ in range(max_rm):
        if not res['converged'] or not res['chol']:           # unobservable: no residual test possible
            break
        x = res['x']
        rows = np.where(mask)[0]
        Hm = g.H(x)[rows]
        w = 1.0 / sig[rows] ** 2
        try:
            Gi = np.linalg.inv(Hm.T @ (w[:, None] * Hm))
        except np.linalg.LinAlgError:
            break
        omega = 1.0 / w - np.einsum('ij,ij->i', Hm @ Gi, Hm)
        r = z[rows] - g.h(x)[rows]
        rn = np.where((omega > 1e-6 / w) & ~g.is_zi[rows], np.abs(r) / np.sqrt(np.maximum(omega, 1e-30)), 0.0)
        j = int(np.argmax(rn))
        if rn[j] < thr:
            break
        mask[rows[j]] = False
        removed += 1
        new = wls(g, z, mask, sig, x0=x)
        if not new['converged'] or not new['chol']:
            mask[rows[j]] = True
            break
        res = new
    res['removed'] = removed
    return res


def errors(g, x_hat, x_true):
    """Per-sample voltage and angle errors (pu, degrees)."""
    dv = x_hat[g.N - 1:] - x_true[g.N - 1:]
    da = np.rad2deg(x_hat[:g.N - 1] - x_true[:g.N - 1])
    return dict(mae_v=float(np.mean(np.abs(dv))), mae_a=float(np.mean(np.abs(da))),
                rmse_v=float(np.sqrt(np.mean(dv ** 2))), rmse_a=float(np.sqrt(np.mean(da ** 2))))
