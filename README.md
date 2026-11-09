# Digital Twin-Assisted AI-Native 6G Edge Intelligence for Privacy-Preserving Federated Learning in Smart Grids

Reference implementation accompanying the manuscript *"Digital Twin-Assisted AI-Native 6G Edge
Intelligence for Privacy-Preserving Federated Learning in Smart Grids"* (submitted to MDPI
*Electronics*, Special Issue "AI-Native Mobile Edge Computing for 6G Networks: Intelligence,
Security, Integration, and Optimization").

## What this is

This repository is a **clean-room implementation of the methodology described in the
manuscript** (Sections 3–7), built so that reviewers and readers can inspect, run, and extend the
pipeline the paper describes: a peer-to-peer federated learning (P2PFL) loop protected with CKKS
homomorphic encryption (HE), orchestrated by a Proximal Policy Optimization (PPO) policy informed
by a Digital Twin (DT) state store, and evaluated against seven baselines, a six-way ablation, a
client-scale sweep, and a simulated-unreliable-client robustness study, exactly as specified in
the paper's equations and experimental design (Sections 3.3–3.6, 4, 5, 6).

It is **not** a verbatim copy of the original lab notebook used to produce the paper's numbers —
that code was not available to port directly. Every component here (the Digital Twin SQLite
store, the GRU forecaster, the simulated 6G/MEC network model, the CKKS P2P aggregation, the trust
score, the multi-objective reward, the PPO/DQN orchestrator and its surrogate training
environment, and the seven-baseline/ablation/scalability/robustness experiment matrix) is
implemented from the paper's equations, algorithm listing, and parameter tables so that the
pipeline is runnable end-to-end and faithful to the described design. **Running this code from
scratch on the real Low Carbon London data will not reproduce the exact numbers in the paper's
tables** (those came from a specific, now-unavailable run with its own random seeds, dataset
partition files, and compute environment) — it reproduces the *method*, for transparency and
reuse, not a byte-for-byte replay of one experiment.

If you are a reviewer checking the paper's claims against runnable code: every equation and
design choice referenced in the paper (Equations 1–5, Algorithm 1, Tables 2–3, Sections 5.1–5.3,
6.4–6.6) has a corresponding, cited implementation below. See [`CODE_DATA_AVAILABILITY.md`](CODE_DATA_AVAILABILITY.md)
for the exact statement intended for the manuscript's Code and Data Availability section.

## Repository layout

```
src/dtppofl/
  config.py            Dataclasses for all hyperparameters, reward weights, simulated ranges
  network_sim.py        Simulated 6G uplink/latency/packet-loss and MEC edge model (Sec. 3.2, Table 3)
  encryption.py          CKKS wrapper (TenSEAL/SEAL, poly modulus 8192, coeff chain 40-21-21-40) (Sec. 3.4, Eq. 2)
  digital_twin.py       SQLite Digital Twin state store, client + round tables (Sec. 4.1, Eq. 3, Table 2)
  trust.py                Trust-score EMA (Sec. 5.2, Eq. 5)
  reward.py               Multi-objective per-round reward (Sec. 3.5, Eq. 4)
  forecaster.py          1-layer GRU load forecaster, per-client z-score windows (Sec. 6.4)
  data.py                  LCL ingestion, cleaning, coverage filter, chronological split,
                            non-IID heterogeneity quantification (Sec. 6.1-6.3)
  aggregation.py        FedAvg (Eq. 1) and round-robin P2P CKKS aggregation (Eq. 2)
  baselines.py            B0-B6 configurations + 5 fixed-strategy selection heuristics (Sec. 6.6, 5.3)
  orchestrator_env.py  Gymnasium surrogate environment for policy training (Sec. 5.1)
  orchestrator.py        PPO / DQN training and frozen-policy inference wrappers (Sec. 5, 4.2)
  fl_experiment.py      Main per-round loop implementing Algorithm 1
  metrics.py               MSE/MAE/RMSE, paired t-test + Cohen's d, results logging (Sec. 6.7)
scripts/
  make_synthetic_lcl_data.py   Generates a small synthetic stand-in dataset so the pipeline runs
                                 out of the box without the real (gated-access) LCL release
  train_orchestrator.py        Trains PPO and DQN on the surrogate env, reproduces Sec. 5.3's comparison
  run_main_comparison.py       B0-B6, 3 seeds (21 runs)
  run_ablation.py              6-way ablation, 2 seeds (12 runs)
  run_scalability.py           30/75/151 clients, 2 seeds (12 runs)
  run_robustness.py            0/10/20/30% unreliable clients, 2 seeds (16 runs)
configs/default.yaml        All default hyperparameters / simulated ranges in one place
tests/                           Smoke tests that exercise each component on synthetic data
```

## Getting the real data

This paper uses the UK Power Networks **Low Carbon London** smart-meter dataset
(CC-BY 4.0, data.london.gov.uk), 22 of its 168 official half-hourly partition files (see Section 7
of the paper for why only 22 of 168). That dataset is not redistributed in this repository —
download your own copy from the London Datastore and point `data.py` at the partition CSVs:

```bash
python -m dtppofl.data build-clients \
    --raw-dir /path/to/lcl_partitions \
    --out-dir data/processed \
    --coverage-threshold 0.90 \
    --window-start 2012-01-01 --window-end 2014-01-31
```

To try the pipeline immediately without the real dataset, generate a small synthetic stand-in
with the same schema (household_id, tariff, timestamp, consumption_kwh):

```bash
python scripts/make_synthetic_lcl_data.py --out-dir data/synthetic --n-households 40
```

## Running the experiment matrix

```bash
pip install -r requirements.txt

# 1. Build the client dataset (real or synthetic — see above)
# 2. Train the orchestrator policies on the fast surrogate environment (Sec. 5.1, 5.3)
python scripts/train_orchestrator.py --config configs/default.yaml --data-dir data/synthetic

# 3. Run the four experiment batches (each logs to results/all_runs.csv)
python scripts/run_main_comparison.py --config configs/default.yaml --data-dir data/synthetic
python scripts/run_ablation.py        --config configs/default.yaml --data-dir data/synthetic
python scripts/run_scalability.py     --config configs/default.yaml --data-dir data/synthetic
python scripts/run_robustness.py      --config configs/default.yaml --data-dir data/synthetic
```

Every run appends one row to `results/all_runs.csv` with the configuration, seed, test
MSE/MAE/RMSE, mean latency/energy/communication bytes, HE overhead, and wall-clock time, matching
the columns referenced throughout Section 7 of the paper.

## Requirements

Python 3.10+, PyTorch, TenSEAL (CKKS), Gymnasium, Stable-Baselines3, NumPy/Pandas/SciPy. See
`requirements.txt`. TenSEAL is optional: if it is not installed, `encryption.py` falls back to a
cost-simulated CKKS stand-in (same timing/payload model, no real cryptography) so the rest of the
pipeline remains runnable in environments where TenSEAL cannot be built; this fallback is clearly
flagged in logs and must not be used to report HE-overhead numbers for any real submission.

## License

Code: MIT License (see `LICENSE`). Dataset: subject to the Low Carbon London dataset's own
CC-BY 4.0 license from UK Power Networks / London Datastore; this repository does not redistribute
it.

## Citation

See `CITATION.cff`.
