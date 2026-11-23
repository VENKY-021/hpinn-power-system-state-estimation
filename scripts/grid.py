"""
Grid model for the H-PINN state-estimation study: MATPOWER/PYPOWER benchmark cases (IEEE 14, 30, 118 bus), AC power flow,
measurement configuration, measurement function h(x), its Jacobian, and operating-scenario generation.
State x = [theta (all buses except the slack), V (all buses)], angles in rad, magnitudes in pu.
"""
import numpy as np
from pypower.api import case14, case30, case118, case300, ppoption
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus
from pypower.makeSbus import makeSbus
from pypower.bustypes import bustypes
from pypower.newtonpf import newtonpf
from pypower.dSbus_dV import dSbus_dV
from pypower.dSbr_dV import dSbr_dV
from pypower.idx_bus import PD, QD, VM, VA, BUS_TYPE
from pypower.idx_gen import PG, QG, VG, GEN_BUS, GEN_STATUS
from pypower.idx_brch import F_BUS, T_BUS, BR_X, BR_R, BR_B

CASES = {'ieee14': case14, 'ieee30': case30, 'ieee118': case118, 'ieee300': case300}
PPOPT = ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10, PF_MAX_IT=30)

# measurement standard deviations (pu, rad)
MEAS = dict(inj=0.85, flow=0.65, pmu=0.10, v_third=3)             # measurement density
REN = dict(pv_frac=0.15, wind_frac=0.10)                               # installed solar / wind capacity (share of base load)
REGIME = dict(n=6, base_weight=0.6, dir_alpha=5.0, load_std=0.20, load_clip=(0.55, 1.5), dv=0.015)   # operating regimes
SIG = {'V': 0.004, 'Pi': 0.010, 'Qi': 0.010, 'Pf': 0.008, 'Qf': 0.008, 'PVm': 0.001, 'PVa': 0.0017, 'ZI': 1e-4}
TYPES = ['V', 'Pi', 'Qi', 'Pf', 'Qf', 'PVm', 'PVa', 'ZI_P', 'ZI_Q']


class Grid:
    def __init__(self, name, seed=7):
        self.name = name
        ppc = ext2int(CASES[name]())
        self.ppc = ppc
        self.baseMVA = ppc['baseMVA']
        self.bus, self.gen, self.branch = ppc['bus'].copy(), ppc['gen'].copy(), ppc['branch'].copy()
        self.N = self.bus.shape[0]
        self.L = self.branch.shape[0]
        self.ref, self.pv, self.pq = bustypes(self.bus, self.gen)
        self.ref = int(self.ref[0])
        self.Ybus, self.Yf, self.Yt = makeYbus(self.baseMVA, self.bus, self.branch)
        self.nonref = np.array([i for i in range(self.N) if i != self.ref])
        self.nx = 2 * self.N - 1
        rng = np.random.default_rng(seed)
        gb = set(self.gen[:, GEN_BUS].astype(int))
        load = (self.bus[:, PD] != 0) | (self.bus[:, QD] != 0)
        # renewable buses: PV at ~15 % and wind at ~10 % of load buses
        lb = np.where(load)[0]
        self.pv_bus = np.sort(rng.choice(lb, max(2, int(round(0.15 * len(lb)))), replace=False))
        rest = np.setdiff1d(lb, self.pv_bus)
        self.wind_bus = np.sort(rng.choice(rest, max(1, int(round(0.10 * len(lb)))), replace=False))
        self.Pd0, self.Qd0 = self.bus[:, PD].copy(), self.bus[:, QD].copy()
        tot = self.Pd0.sum()
        self.pv_cap = REN['pv_frac'] * tot / len(self.pv_bus)
        self.wind_cap = REN['wind_frac'] * tot / len(self.wind_bus)
        self.Pg0 = self.gen[:, PG].copy()
        self.x_nom = self.branch[:, BR_X].copy()
        self.r_nom = self.branch[:, BR_R].copy()
        # zero-injection buses: no load, no generator
        self.zi = np.array([i for i in range(self.N) if i not in gb and not load[i]], dtype=int)
        nzi = np.array([i for i in range(self.N) if i not in set(self.zi)])
        # SCADA voltage magnitudes: generator buses and every third remaining bus
        others = [i for i in range(self.N) if i not in gb]
        self.mV = np.array(sorted(gb | set(others[::MEAS['v_third']])), dtype=int)
        # injection pairs at ~85 % of the non-zero-injection buses
        k = int(round(MEAS['inj'] * len(nzi)))
        self.mI = np.sort(rng.choice(nzi, k, replace=False))
        # branch-flow pairs (from end) at ~65 % of branches
        self.mF = np.sort(rng.choice(self.L, int(round(MEAS['flow'] * self.L)), replace=False))
        # PMUs (voltage phasor) at ~10 % of buses, at least 2, never at the slack
        npmu = max(2, int(round(MEAS['pmu'] * self.N)))
        self.mP = np.sort(rng.choice(self.nonref, npmu, replace=False))
        self.build_index()
        self.n_added = self.ensure_observable()
        # operating regimes (dispatch shares, regional load pattern, voltage set-point offsets); regime R-1 is held out for testing
        self.R = REGIME['n']
        ng = self.gen.shape[0]
        active = np.where((self.Pg0 > 0) & (self.gen[:, GEN_BUS].astype(int) != self.ref))[0]
        self.share = np.zeros((self.R, ng))
        base = self.Pg0[active] / self.Pg0[active].sum()
        for r in range(self.R):
            self.share[r, active] = REGIME['base_weight'] * base + (1 - REGIME['base_weight']) * rng.dirichlet(np.full(len(active), REGIME['dir_alpha']))
        self.load_pat = np.clip(1 + REGIME['load_std'] * rng.standard_normal((self.R, self.N)), *REGIME['load_clip'])
        self.dv = rng.uniform(-REGIME['dv'], REGIME['dv'], size=(self.R, ng))

    def ensure_observable(self, tol=1e-6, max_add=200):
        """Design step: add meters near the null space until the complete measurement set is numerically observable."""
        Va, Vm, ok, _ = self.solve_pf(1.0, np.ones(self.N), 0.4, 0.4)
        x = self.x_from(Va, Vm)
        fb, tb = self.branch[:, F_BUS].astype(int), self.branch[:, T_BUS].astype(int)
        for it in range(max_add):
            H = self.H(x)
            A = H / np.linalg.norm(H, axis=1, keepdims=True).clip(1e-12)
            s, Vt = np.linalg.svd(A, full_matrices=False)[1:]
            if s[-1] / s[0] > tol:
                return it
            j = int(np.argmax(np.abs(Vt[-1])))
            bus = int(self.nonref[j]) if j < self.N - 1 else j - (self.N - 1)
            if bus not in set(self.mI) and bus not in set(self.zi):
                self.mI = np.sort(np.append(self.mI, bus))
            else:
                inc = [l for l in range(self.L) if (fb[l] == bus or tb[l] == bus) and l not in set(self.mF)]
                if inc:
                    self.mF = np.sort(np.append(self.mF, inc[0]))
                elif bus not in set(self.mV):
                    self.mV = np.sort(np.append(self.mV, bus))
                elif bus not in set(self.mP) and bus != self.ref:
                    self.mP = np.sort(np.append(self.mP, bus))
                else:
                    raise RuntimeError('cannot make the measurement set observable')
            self.build_index()
        raise RuntimeError('observability design did not converge')

    def build_index(self):
        parts = [('V', self.mV), ('Pi', self.mI), ('Qi', self.mI), ('Pf', self.mF), ('Qf', self.mF),
                 ('PVm', self.mP), ('PVa', self.mP), ('ZI_P', self.zi), ('ZI_Q', self.zi)]
        self.mtype, self.mloc = [], []
        for t, idx in parts:
            self.mtype += [t] * len(idx)
            self.mloc += list(idx)
        self.mtype = np.array(self.mtype)
        self.mloc = np.array(self.mloc, dtype=int)
        self.M = len(self.mtype)
        self.sigma = np.array([SIG['ZI'] if t.startswith('ZI') else SIG[t] for t in self.mtype])
        self.is_zi = np.array([t.startswith('ZI') for t in self.mtype])
        # bus that hosts each meter (for RTU outages): flows hosted at the from bus
        fb = self.branch[:, F_BUS].astype(int)
        self.mhost = np.array([fb[l] if t in ('Pf', 'Qf') else l for t, l in zip(self.mtype, self.mloc)])
        self.sl = {t: np.where(self.mtype == t)[0] for t in TYPES}

    # ---------------------------------------------------------------- state helpers
    def x_from(self, Va, Vm):
        return np.concatenate([Va[self.nonref], Vm])

    def V_from(self, x):
        Va = np.zeros(self.N)
        Va[self.nonref] = x[:self.N - 1]
        Vm = x[self.N - 1:]
        return Vm * np.exp(1j * Va)

    # ---------------------------------------------------------------- measurement model
    def h(self, x):
        V = self.V_from(x)
        S = V * np.conj(self.Ybus @ V)
        Sf = V[self.branch[:, F_BUS].astype(int)] * np.conj(self.Yf @ V)
        Vm, Va = np.abs(V), np.angle(V)
        return np.concatenate([Vm[self.mV], S.real[self.mI], S.imag[self.mI], Sf.real[self.mF], Sf.imag[self.mF],
                               Vm[self.mP], Va[self.mP], S.real[self.zi], S.imag[self.zi]])

    def H(self, x):
        V = self.V_from(x)
        dS_dVm, dS_dVa = dSbus_dV(self.Ybus, V)
        dSf_dVa, dSf_dVm, _, _, _, _ = dSbr_dV(self.branch, self.Yf, self.Yt, V)
        dS_dVa, dS_dVm = dS_dVa.toarray(), dS_dVm.toarray()
        dSf_dVa, dSf_dVm = dSf_dVa.toarray(), dSf_dVm.toarray()
        N, nr = self.N, self.nonref
        I = np.eye(N)
        rows = []
        def blk(Aa, Am):
            return np.hstack([Aa[:, nr], Am])
        rows.append(blk(np.zeros((len(self.mV), N)), I[self.mV]))
        rows.append(blk(dS_dVa.real[self.mI], dS_dVm.real[self.mI]))
        rows.append(blk(dS_dVa.imag[self.mI], dS_dVm.imag[self.mI]))
        rows.append(blk(dSf_dVa.real[self.mF], dSf_dVm.real[self.mF]))
        rows.append(blk(dSf_dVa.imag[self.mF], dSf_dVm.imag[self.mF]))
        rows.append(blk(np.zeros((len(self.mP), N)), I[self.mP]))
        rows.append(blk(I[self.mP], np.zeros((len(self.mP), N))))
        rows.append(blk(dS_dVa.real[self.zi], dS_dVm.real[self.zi]))
        rows.append(blk(dS_dVa.imag[self.zi], dS_dVm.imag[self.zi]))
        return np.vstack(rows)

    # ---------------------------------------------------------------- power flow
    def solve_pf(self, load_scale, bus_noise, a_pv, a_wind, xpert=None, regime=None, share_noise=None):
        """Return (Va, Vm, ok, branch) for one operating point. bus_noise: per-bus multiplicative load factor.
        regime=None: generation follows the base case proportionally; regime=r: regime dispatch shares, load pattern and
        voltage set-point offsets."""
        bus, gen, branch = self.bus.copy(), self.gen.copy(), self.branch.copy()
        f = bus_noise if regime is None else bus_noise * self.load_pat[regime]
        Pd = self.Pd0 * load_scale * f
        Qd = self.Qd0 * load_scale * f
        Pd[self.pv_bus] -= self.pv_cap * a_pv
        Pd[self.wind_bus] -= self.wind_cap * a_wind
        bus[:, PD], bus[:, QD] = Pd, Qd
        net = Pd.sum()
        if regime is None:
            gen[:, PG] = self.Pg0 * (net / self.Pd0.sum())
        else:
            sh = self.share[regime] * (1.0 if share_noise is None else share_noise)
            gen[:, PG] = 1.03 * net * sh / sh.sum()
            gen[:, VG] = gen[:, VG] + self.dv[regime]
        Ybus = self.Ybus
        if xpert is not None:
            branch[:, BR_X] = self.x_nom * xpert
            branch[:, BR_R] = self.r_nom * xpert
            Ybus, _, _ = makeYbus(self.baseMVA, bus, branch)
        Sbus = makeSbus(self.baseMVA, bus, gen)
        V0 = bus[:, VM] * np.exp(1j * np.deg2rad(bus[:, VA]))
        on = gen[:, GEN_STATUS] > 0
        gbus = gen[on, GEN_BUS].astype(int)
        V0[gbus] = gen[on, VG] / np.abs(V0[gbus]) * V0[gbus]
        V, ok, _ = newtonpf(Ybus, Sbus, V0, np.array([self.ref]), self.pv, self.pq, PPOPT)
        Va = np.angle(V)
        Va = np.angle(np.exp(1j * (Va - Va[self.ref])))      # slack angle is the reference (0)
        return Va, np.abs(V), bool(ok), branch if xpert is not None else None

    def true_meas(self, Va, Vm, branch=None):
        """Exact measurement vector for a power-flow solution (with the true, possibly perturbed, network)."""
        if branch is None:
            return self.h(self.x_from(Va, Vm))
        Ybus, Yf, _ = makeYbus(self.baseMVA, self.bus, branch)
        V = Vm * np.exp(1j * Va)
        S = V * np.conj(Ybus @ V)
        Sf = V[self.branch[:, F_BUS].astype(int)] * np.conj(Yf @ V)
        return np.concatenate([Vm[self.mV], S.real[self.mI], S.imag[self.mI], Sf.real[self.mF], Sf.imag[self.mF],
                               Vm[self.mP], Va[self.mP], S.real[self.zi], S.imag[self.zi]])

    def neighbours(self, b, k):
        """k buses closest to bus b in hops (including b) for an RTU / communication-area outage."""
        fb, tb = self.branch[:, F_BUS].astype(int), self.branch[:, T_BUS].astype(int)
        adj = {i: set() for i in range(self.N)}
        for f, t in zip(fb, tb):
            adj[f].add(t); adj[t].add(f)
        seen, front = [b], [b]
        while len(seen) < k and front:
            nxt = []
            for u in front:
                for v in sorted(adj[u]):
                    if v not in seen and len(seen) < k:
                        seen.append(v); nxt.append(v)
            front = nxt
        return np.array(seen)


def describe(g):
    return dict(name=g.name, N=g.N, L=g.L, M=int(g.M), nx=g.nx, nV=len(g.mV), nI=len(g.mI), nF=len(g.mF),
                nPMU=len(g.mP), nZI=len(g.zi), redundancy=round(g.M / g.nx, 2), nPV=len(g.pv_bus), nW=len(g.wind_bus))


if __name__ == '__main__':
    for n in CASES:
        g = Grid(n)
        Va, Vm, ok, _ = g.solve_pf(1.0, np.ones(g.N), 0.5, 0.5)
        x = g.x_from(Va, Vm)
        # Jacobian check by finite differences
        Hn = g.H(x)
        e = 1e-6
        Hfd = np.column_stack([(g.h(x + e * np.eye(g.nx)[k]) - g.h(x - e * np.eye(g.nx)[k])) / (2 * e) for k in range(g.nx)])
        print(describe(g), 'pf ok', ok, 'max|H-Hfd|', np.abs(Hn - Hfd).max(), 'ZI residual', np.abs(g.h(x)[g.is_zi]).max())
