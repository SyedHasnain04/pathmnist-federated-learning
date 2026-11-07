# DEVIATIONS.md

## Deviations from Specification

This file records every deliberate or forced deviation from the `PathMNIST_FedAvg_FedProx_Final_Execution_Plan.md` specification, with justification and impact assessment.

---

### DEV-01: W&B Online Logging Unavailable — Local JSON Fallback

**Specification requirement:** Use W&B project `pathmnist-federated-learning` for all logging.

**Actual behavior:** W&B is completely disabled. All metrics are saved as local JSON files at `results/{run_name}/history.json`.

**Root cause:** The sandbox environment restricts network access to W&B servers. Additionally, `wandb init` in offline mode fails on Windows due to `PermissionError: mkdir AppData\Local\wandb: Access is denied` and Unix domain socket limitations.

**Fallback:** Comprehensive local JSON logging captures all required metrics:
- Per-round: `official_val_accuracy`, `official_val_loss`, `official_test_accuracy`, `official_test_loss`, `duration_seconds`
- Per-client: `client_holdout_accuracies`, `client_holdout_losses`, `client_train_losses`
- Update metrics: `update_norms`, `standard_cosines`, `leave_one_out_cosines`, `global_update_norm`
- Config: full hyperparameter snapshot saved in `history.json["config"]`

**Impact:** No public W&B dashboard available. All data for analysis and reproduction is preserved locally. Plots and tables are generated directly from `history.json` files. Logs can be uploaded to W&B offline sync later if network access becomes available.

**Detectability:** Anyone attempting to reproduce via W&B will find no runs. The local JSON logs are the authoritative record.

---

### DEV-02: PerturbedGradientDescent Has No Momentum — PyTorch SGD Used for FedProx

**Specification requirement:** Use PyTorch SGD with `momentum=0.9` for both FedAvg and FedProx.

**Actual behavior:** PyTorch `torch.optim.SGD(momentum=0.9)` is used for both algorithms.

**Root cause:** PFLlib's `PerturbedGradientDescent` optimizer (at `PFLlib/system/flcore/optimizers/fedoptimizer.py`) implements the proximal gradient step by modifying parameter gradients manually, but does **not** include a momentum buffer. It explicitly sets `lr` and `mu` but has no momentum parameter.

**Fallback:** Per specification mandate, the proximal loss is computed as `(mu/2) * sum ||w_k - w_global||^2`, added to the cross-entropy loss, and minimized via standard `torch.optim.SGD(momentum=0.9)`. This is the correct and intended implementation per the execution plan (which explicitly states "do NOT use PerturbedGradientDescent").

**Impact:** None — the execution plan explicitly required PyTorch SGD. The μ=0 equivalence test confirms FedProx with μ=0 produces identical results to FedAvg (max parameter difference < 1e-7).

---

### DEV-03: Compute Environment Transition — Execution on CUDA GPU (Google Colab)

**Specification requirement:** Rigorous benchmark across all 12 experimental conditions and preliminary μ sweep.

**Decision & Execution:** Initial local profiling on a 22-core CPU indicated long communication round durations (~3.5–5 minutes per round), which would have required ~40–50 wall-clock hours for 15 runs. Per strict scientific protocol, **all partial CPU runs were halted and discarded** to guarantee zero splicing of CPU and GPU runs within or across experimental groups. The entire experimental matrix (from the μ sweep onward) is executed uniformly on CUDA GPU hardware in Google Colab.

**Impact on Results:** Scientifically pure comparison. Every single run in the final results table, plots, and analysis originates from an identical CUDA compute environment under identical precision and device semantics. Per-round duration drops to ~8–12 seconds, allowing complete 100-round convergence across all 15 runs within ~2.5–3 hours.

---

### DEV-04: PFLlib Dataset Generator Bypassed

**Specification requirement:** Use official PathMNIST splits.

**PFLlib behavior found:** `generate_MNIST.py` and related generators in `PFLlib/system/flcore/trainmodel/` merge the official train and test splits into a single dataset before partitioning, violating the train/test separation requirement.

**Actual behavior:** A fully custom data pipeline (`generate_pathmnist_data.py`) was implemented that:
1. Loads the official `pathmnist.npz` directly from medmnist
2. Preserves official train (89,996), val (10,004), test (7,180) splits exactly
3. Applies Dirichlet partitioning only to the official training set
4. Saves val and test as separate read-only official evaluation sets
5. Verifies split integrity with assertions before any partitioning

**Impact:** Positive — this deviation strengthens scientific validity by preserving the required official split integrity. The specification explicitly mandates this separation.

---

### DEV-05: Checkpoint Interval Changed from 25 to 10 Rounds

**Specification requirement:** Not explicitly specified.

**Actual behavior:** Intermediate `history.json` and `checkpoint_model.pt` saved every 10 rounds instead of 25.

**Root cause:** With 8-10 hour runs on CPU, more frequent checkpointing (every ~45 min) reduces data loss if the process is interrupted.

**Impact:** Positive — improves fault tolerance. No scientific impact.

---

### DEV-06: FedProx Noisy Seed 42 Sweep Run — Reuse Eligibility

**Status:** Pending μ sweep completion.

**Specification requirement:** The noisy seed 42 FedProx sweep run may be reused as `fedprox_noisy_seed42` only if config, code, artifact hashes, and all 100 rounds are identical to final spec.

**Decision criteria:** After sweep completes, verify:
1. Same initial checkpoint (`initial_model_seed_42.pt`)
2. Same data partitions (SHA256 hashes)
3. Same hyperparameters
4. All 100 rounds complete and logged

If all conditions met, the best-μ sweep run is designated `fedprox_noisy_seed42` without rerunning. Otherwise, a separate `fedprox_noisy_seed42` run is launched.

**Update when resolved.**

---

## Tests Passing

| Test | Status | Description |
|------|--------|-------------|
| `test_data_integrity.py::test_official_split_integrity` | ✅ PASS | Official splits preserved, no leakage |
| `test_data_integrity.py::test_noise_integrity` | ✅ PASS | 40% noise applied exactly to Client 0 only |
| `test_data_integrity.py::test_model_architecture` | ✅ PASS | FedAvgCNN output shape (1,9), 583,113 params |
| `test_optimizer_and_mu_equivalence.py::test_fresh_optimizer` | ✅ PASS | Optimizer state fully reset each round |
| `test_optimizer_and_mu_equivalence.py::test_mu0_equivalence` | ✅ PASS | FedProx μ=0 ≡ FedAvg (max diff < 1e-7) |
| `test_optimizer_and_mu_equivalence.py::test_proximal_constraint` | ✅ PASS | Proximal term uses start-of-round snapshot |
| `test_metrics_evaluator.py::test_evaluator_immutability` | ✅ PASS | Evaluation doesn't modify model parameters |
| `test_metrics_evaluator.py::test_zero_norm_handling` | ✅ PASS | Zero-norm clients return NaN cosines |

---

## Non-Deviations (Specification Complied With)

The following critical requirements were verified and complied with:

- ✅ Official train/val/test splits never merged
- ✅ Official val used for μ selection; official test used only for final evaluation
- ✅ 5-client Dirichlet (α=0.3) partitioning with seeded 75:25 local split
- ✅ Client 0 is always the noisy client
- ✅ Exactly 40% symmetric label noise (using `round(0.4 * n0)` with Python ties-to-even)
- ✅ Clean and noisy artifacts saved independently and identically for clients 1-4
- ✅ Identical initialization checkpoint for all 4 groups per seed
- ✅ FedAvg and FedProx use identical SGD (lr=0.01, momentum=0.9, no weight decay)
- ✅ Fresh optimizer every round (no momentum carry-over between rounds)
- ✅ FedProx proximal term against detached start-of-round global snapshot
- ✅ μ=0 equivalence test passed
- ✅ All 5 clients participate every round (join_ratio=1.0, no dropout)
- ✅ Sample-count-weighted aggregation
- ✅ Update metrics (L2 norms, standard cosine, LOO cosine) computed before aggregation
- ✅ Client-local holdout, official val, official test evaluations separate and non-overlapping
- ✅ Minimum client size ≥ 1,000 verified for all seeds (see partition metadata)

