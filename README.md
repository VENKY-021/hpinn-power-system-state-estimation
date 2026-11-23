# Simulation data: hybrid physics-informed state estimation on IEEE 14/30/118-bus systems

AC power-system state estimation under measurement uncertainty, simulated on the IEEE 14-, 30- and 118-bus benchmark cases
(PYPOWER/MATPOWER data). This repository holds the **simulation code and the simulation results**. Every number is produced by the scripts
here and stored in files, nothing is typed by hand.

## What was simulated

- **Systems:** IEEE 14, 30 and 118 bus (a 300-bus case for the scaling test) and a unimodal 30-bus control case (`u_ieee30`).
- **22 test scenarios per system** (20 for the control case): nominal, noise level 0.5x to 4x, heavy-tailed contamination, missing measurements 5-50 %,
  a lost area, systematic bias 3/6/10 %, line-parameter errors 2/5/10 %, an unseen operating regime (`shift`, `shift_miss30`) and a mixed case.
  300 / 300 / 200 test snapshots per scenario (14 / 30 / 118 bus).
- **Estimators compared:** WLS, robust WLS, WLS with normalised-residual test, empirical-prior WLS/RWLS (EP), DNN, PINN, DNN + Gauss-Newton,
  an ensemble of DNNs, and the proposed **H-PINN** (physics-informed network + robust MAP correction with a fused prior and an evidence-based safeguard),
  each with split-conformal intervals per observability class.
- Fixed seeds; networks are averaged over training seeds.

## Highlights (from `results/summary.json`)

Mean voltage-magnitude error at nominal conditions, in p.u.:

| System | WLS | EP-RWLS + safeguard | H-PINN (proposed) | vs WLS | vs EP-RWLS |
|---|---|---|---|---|---|
| IEEE 14 | 0.00109 | 0.00058 | **0.00034** | -68 % | -41 % |
| IEEE 30 | 0.00114 | 0.00053 | **0.00033** | -71 % | -39 % |
| IEEE 118 | 0.00066 | 0.00022 | **0.00012** | -81 % | -45 % |

- The proposed estimator has the lowest voltage error among the benchmark estimators in most scenarios on all three systems
  and its angle error stays within 3 % of the best estimator in nearly all of them (`scripts/analysis.py` computes the counts).
- Label efficiency (`results/label_eff.json`): the physics-informed network needs fewer labelled snapshots than the plain DNN to reach the same error.
- Cost (`results/timing.json`, `results/scaling.json`): a few times the runtime of WLS, still in the millisecond range up to 118 buses.
- **Limits that the data show:** under an unseen operating regime WLS remains the best voltage estimator on the 14- and 30-bus systems
  (the safeguard recovers the 118-bus case), and the conformal intervals lose coverage when the line parameters are wrong by 10 %.
  Gains over the empirical-prior baseline are smaller on the unimodal control case.

## Files

- `results/eval/<system>/<scenario>.npz` raw per-snapshot results of every estimator. Keys are `<method>/<metric>`
  (`mae_v`, `mae_a`, `rmse_*`, `iters`, `ok`, interval coverage and width). Proposed method = `hyev:*` (analysis name `hyep`),
  H-PINN without safeguard = `hyep:*` (`hyep0`), EP-RWLS + safeguard = `epev`. `*_hp.json` hold the tuned hyper-parameters.
- `results/summary.json` per system and scenario means of all estimators (the single source of the numbers above).
- `results/label_eff.json`, `timing.json`, `scaling.json`, `consistency.json`, `noise_scale.json`, `shift_scale.json` additional studies.
- `results/*.log` training and evaluation logs.
- `scripts/` the code: `grid.py` (grid model, measurement function, scenarios), `data.py`/`build_data.py` (data generation),
  `estimators.py` (WLS, RWLS, EP, H-PINN correction, safeguard), `nn.py`/`train_all.py` (networks), `evaluate.py` (all scenarios),
  `analysis.py`, `export_summary.py`, and the study scripts `label_eff.py`, `timing.py`, `scaling.py`, `consistency_check.py`,
  `noise_scale_check.py`, `shift_scale_check.py`.

Not included because of size: the generated training/test sets (regenerate with `scripts/build_data.py`) and the trained network weights (`scripts/train_all.py`).

Python 3 with numpy, scipy, PyPOWER and PyTorch.
