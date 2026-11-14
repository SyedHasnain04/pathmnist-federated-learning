# PathMNIST Federated Learning: FedAvg vs FedProx under Non-IID Data and Label Noise

Links: [GitHub repository](https://github.com/syedHasnain04/pathmnist-federated-learning) | [W&B project](https://wandb.ai/iamsyedhasnain04-gprec/pathmnist-federated-learning) | One-page summary: `submission/summary.pdf`

## 1. Objective

Compare FedAvg and FedProx on PathMNIST with 5 clients, Dirichlet(alpha = 0.3) non-IID partitions, and 40% label noise on one client (Client 0). Both algorithms use identical partitions, noise assignments, initial weights, and training settings per seed. The goal is to measure the difference, not to assume FedProx wins.

### Experimental design (2 x 2, 3 seeds = 12 final runs)

| Algorithm | Clean labels | 40% noise on Client 0 |
|---|---|---|
| FedAvg | `fedavg_clean_seed{42,123,2026}` | `fedavg_noisy_seed{42,123,2026}` |
| FedProx | `fedprox_clean_seed{42,123,2026}` | `fedprox_noisy_seed{42,123,2026}` |

A mu sweep (mu in {0.001, 0.01, 0.1}; noisy setting, seed 42) selects the FedProx coefficient. The winning sweep run is reused as `fedprox_noisy_seed42`, so 14 trainings produce 12 final runs plus 2 non-selected sweep runs. All runs are logged to the W&B project above.

## 2. Dataset and split integrity

- PathMNIST (MedMNIST): 28x28 RGB histopathology images, 9 classes.
- Official splits are preserved: train 89,996, validation 10,004, test 7,180.
- Only the official **train** split is partitioned among clients. The official **validation** split is used for model selection (mu) and in-distribution tracking. The official **test** split is used only for global evaluation.
- PFLlib's dataset generator template (`generate_MNIST.py`) merges train and test before partitioning. This project does **not** use that behavior; `generate_pathmnist_data.py` partitions the official train split only.
- Images are normalized per channel. Official validation/test files are hashed (SHA-256) and the hashes are checked by tests.

## 3. Partitioning and label noise

**Partition.** Class-wise Dirichlet(alpha = 0.3) over the official train split into 5 clients, with a minimum of 1,000 samples per client (pre-split). Each client's data is then split 75:25 into local train and local held-out test, deterministically per seed. All partitions are generated once per seed, saved to disk, and loaded by all four groups.

**Noise.** Client 0 only, training labels only. Exactly `round(0.4 * n0)` samples are chosen; each gets a replacement label drawn uniformly from the other 8 classes. Client 0's local test labels, the official validation labels, and the official test labels are never modified. Noise indices and original/replacement labels are saved. Clean groups use the original labels at the same indices.

### Client sizes and Client 0 aggregation weight

| Seed | Client 0 train (noisy) | Client 1 | Client 2 | Client 3 | Client 4 | Total train | C0 weight p0 |
|---|---|---|---|---|---|---|---|
| 42 | 7,910 (3,164) | 15,460 | 12,981 | 16,113 | 15,032 | 67,496 | 0.1172 |
| 123 | 23,101 (9,240) | 15,060 | 2,297 | 14,355 | 12,683 | 67,496 | 0.3423 |
| 2026 | 8,874 (3,550) | 6,415 | 20,474 | 14,643 | 17,090 | 67,496 | 0.1315 |

Client 0's aggregation weight differs about three-fold across seeds, so the noisy client has far more influence on seed 123. Results are therefore reported per seed as well as mean +/- std.

## 4. Model and training configuration

| Item | Value |
|---|---|
| Model | PFLlib `FedAvgCNN`, `in_features=3`, `num_classes=9`, `dim=1024` |
| Layers | Conv(3->32, 5x5) - ReLU - MaxPool2, Conv(32->64, 5x5) - ReLU - MaxPool2, FC 1024->512, FC 512->9 |
| Parameters | 583,113 (no BatchNorm) |
| Rounds / local epochs / batch size | 100 / 5 / 64 |
| Optimizer | SGD, lr 0.01, momentum 0.9, weight decay 0 |
| Participation | 100% (`join_ratio = 1.0`) |
| Aggregation | Sample-count weighted |
| Initialization | One initial checkpoint per seed, SHA-256 checked against `checkpoints/initial/manifest.json`, loaded by all four groups |

<!-- VERIFY: layer description matches the model code; the parameter count 583,113 corresponds to 32/64 filters, 5x5 kernels, no padding -->

## 5. Algorithms

**FedAvg.** Clients train locally; the server averages client weights, weighted by local training sample count.

**FedProx.** Local objective `F_k(w) + (mu/2) * ||w - w_t||^2`, where `w_t` is a detached copy of the global model taken at the start of the round. The penalty is added to the local loss, so it enters the gradient. A fresh SGD optimizer is created every round, so momentum is never carried across rounds. With mu = 0, FedProx matches FedAvg numerically (unit test, max parameter difference below 1e-7 on CPU).

**mu selection.** Highest mean official validation accuracy over rounds 91-100 on the noisy seed-42 run; ties go to the smaller mu. Selected: **mu = 0.1**.

## 6. Update metrics

Computed every round from flattened trainable parameters, before aggregation:

- Update: `Delta_k = w_k_local - w_global`; L2 norm `||Delta_k||_2`.
- Aggregated update: `Delta_global = sum_k p_k * Delta_k`, with `p_k = n_k / sum_j n_j`.
- Standard cosine: `cos(Delta_k, Delta_global)`.
- Leave-one-out cosine: `cos(Delta_k, Delta_-k)`, where `Delta_-k` is the sample-weighted mean of the other four client updates. This removes the bias from the client being part of the aggregate it is compared against.
- Cosine is logged as undefined if either norm is zero.

## 7. Evaluation

| Evaluation | Data | Meaning |
|---|---|---|
| Global test accuracy/loss (every round) | Official test (7,180) | **Cross-site** generalization |
| Validation accuracy/loss (every round) | Official validation (10,004) | In-distribution |
| Per-client accuracy/loss (every round) | Each client's local held-out 25% | In-distribution, client-specific |
| Confusion matrix | Official test, final round | |

**Distribution shift.** The official PathMNIST test set comes from a different clinical site than train/validation. Cross-site test accuracy is therefore lower than in-distribution accuracy. This is expected and not an implementation bug. Validation and client holdout metrics should not be compared directly with the cross-site test metric.

Reported numbers: round 100 and the mean of rounds 91-100 (robustness to round-to-round oscillation).

## 8. Results

All runs: 100 rounds, NVIDIA Tesla T4 (recorded per run in `history.json` under `env.gpu_name`). <!-- VERIFY: GPU name -->

### mu sweep (FedProx, noisy, seed 42)

| mu | Val acc (mean 91-100) | Test acc (mean 91-100) | Val acc (round 100) | Selected |
|---|---|---|---|---|
| 0.001 | 0.8170 | 0.7383 | 0.8172 | |
| 0.01 | 0.8275 | 0.7456 | 0.8283 | |
| 0.1 | 0.8455 | 0.7538 | 0.8464 | yes |

Validation accuracy rose monotonically with mu and the best value is at the edge of the grid; larger mu values were not tested.

### Cross-site test accuracy (mean of rounds 91-100)

| Group | Seed 42 (p0=0.117) | Seed 123 (p0=0.342) | Seed 2026 (p0=0.132) | Mean +/- std |
|---|---|---|---|---|
| FedAvg clean | 0.7749 | 0.7629 | 0.7768 | 0.7715 +/- 0.0061 |
| FedAvg noisy | 0.7491 | 0.6500 | 0.7457 | 0.7149 +/- 0.0459 |
| FedProx clean | 0.7706 | 0.7645 | 0.7667 | 0.7673 +/- 0.0025 |
| FedProx noisy | 0.7538 | 0.6880 | 0.7289 | 0.7236 +/- 0.0271 |

### Validation accuracy, in-distribution (mean of rounds 91-100)

| Group | Seed 42 | Seed 123 | Seed 2026 | Mean +/- std |
|---|---|---|---|---|
| FedAvg clean | 0.8510 | 0.8488 | 0.8467 | 0.8488 +/- 0.0017 |
| FedAvg noisy | 0.8180 | 0.6740 | 0.7945 | 0.7622 +/- 0.0631 |
| FedProx clean | 0.8645 | 0.8433 | 0.8631 | 0.8570 +/- 0.0097 |
| FedProx noisy | 0.8455 | 0.7920 | 0.8370 | 0.8248 +/- 0.0234 |

### Per-client and round-100 results

Round-100 test accuracy, per-client holdout accuracy, and all per-run values are in `submission/metrics/results.md` and `submission/metrics/all_results.json`. Client 0 clean-holdout accuracy at round 100 (mean +/- std): FedAvg clean 0.878 +/- 0.034, FedAvg noisy 0.751 +/- 0.064, FedProx clean 0.885 +/- 0.023, FedProx noisy 0.829 +/- 0.013.

### Derived comparisons (from the tables above)

| Comparison | Seed 42 | Seed 123 | Seed 2026 | Mean |
|---|---|---|---|---|
| FedAvg test drop from noise (clean - noisy) | 0.0258 | 0.1129 | 0.0311 | 0.0566 |
| FedProx test drop from noise | 0.0168 | 0.0765 | 0.0378 | 0.0437 |
| FedAvg val drop from noise | 0.0330 | 0.1748 | 0.0522 | 0.0866 |
| FedProx val drop from noise | 0.0190 | 0.0513 | 0.0261 | 0.0321 |
| Noisy: FedProx - FedAvg, test | +0.0047 | +0.0380 | -0.0168 | +0.0087 |
| Noisy: FedProx - FedAvg, val | +0.0275 | +0.1180 | +0.0425 | +0.0627 |

Plots (all in `submission/plots/`): cross-site test accuracy and loss, per-client accuracy, validation accuracy, update L2 norms, standard cosine, leave-one-out cosine, client class-distribution heatmap, aggregation weights, mu sweep.

## 9. Analysis

<!-- TODO: edit after checking the update-norm and cosine plots for Client 0 -->

<!--**Draft (edit to match the plots):** -->

Under 40% label noise on Client 0, FedAvg's cross-site test accuracy fell from 0.772 to 0.715 (mean of rounds 91-100). The drop was largest on seed 123 (0.113), where Client 0 carries the largest aggregation weight (p0 = 0.342), and smaller on seeds 42 and 2026 (0.026 and 0.031).

FedProx (mu = 0.1) reduced the in-distribution damage from noise in all three seeds: the validation-accuracy drop averaged 0.032 for FedProx versus 0.087 for FedAvg, and Client 0's clean-holdout accuracy was higher (0.829 vs 0.751 at round 100). On the clean setting, FedProx and FedAvg were nearly equal on cross-site test accuracy (0.767 vs 0.772) and differed by 0.008 on validation accuracy.

On the cross-site test set, FedProx's advantage under noise was inconsistent: +0.005, +0.038, and -0.017 on seeds 42, 123, and 2026 (mean +0.009), which is smaller than the across-seed standard deviation (0.027 to 0.046). With three seeds, no significance claim is made. The data are consistent with FedProx mitigating the in-distribution effect of the noisy client but do not show a reliable cross-site improvement.



## 10. Limitations

1. Three seeds give weak variability estimates; no significance tests are reported.
2. Client 0's aggregation weight varies from 0.117 to 0.342 across seeds, which dominates the variance of the noisy groups. Seed 123 carries most of the noise effect.
3. mu was chosen on one seed (42) using in-distribution validation accuracy, which may not predict cross-site performance. The selected mu = 0.1 is at the edge of the swept grid.
4. A stronger proximal term restricts all local drift, not only noise-driven drift. The small gain on clean data argues against a pure regularization effect but does not rule it out.
5. SGD momentum (0.9) changes the practical effect of the proximal term, so mu is not directly comparable to values from the original FedProx paper.
6. Cross-site test accuracy reflects a clinical-site distribution shift. Per-client holdout sets are small and non-IID, so per-client accuracies are noisy.
7. Runs used a CUDA GPU; results are not bit-identical to CPU runs. On the Colab T4, 10 of 11 tests passed; `test_mu_zero_equivalence_cuda_tolerant` failed at its 1e-5 tolerance (max difference 1.09e-4). <!-- UPDATE: replace with the diagnosed noise floor / fixed test result -->

## 11. PFLlib usage and modifications

This project uses the `FedAvgCNN` architecture from [PFLlib](https://github.com/TsingZ0/PFLlib) (Apache License 2.0; the license is kept at `PFLlib/LICENSE`). The training loop, FedAvg/FedProx logic, data pipeline, and metrics are implemented in this repository's own code (`federated_runner.py`, `generate_pathmnist_data.py`, `metrics_evaluator.py`) rather than by running PFLlib's servers and clients. Differences from PFLlib:

1. **Data pipeline.** PFLlib's generator merges official train and test before partitioning. Replaced with a generator that partitions only the official train split and keeps validation and test separate.
2. **Optimizer.** PFLlib's FedProx client uses `PerturbedGradientDescent`, which has no momentum. This project uses `torch.optim.SGD` (momentum 0.9) with an explicit proximal loss term, and a fresh optimizer each round.
3. **Proximal reference.** The round-start global parameters are cloned and detached once per round and used as the fixed reference. <!-- VERIFY: confirm against your PFLlib audit notes in DEVIATIONS.md; do not claim upstream bugs you did not observe -->
4. **Diagnostics.** Per-client update norms and standard/leave-one-out cosine similarities are computed from local weights before aggregation.
5. **Initialization.** One initial checkpoint per seed with SHA-256 verification, shared by all four groups.

See `DEVIATIONS.md` for the full record.

## 12. Reproducing

Tests: `python -m pytest tests/ -v` (11 tests: data integrity, noise integrity, model dimensions, init-hash equality, evaluator correctness, update metrics, optimizer lifecycle, mu = 0 equivalence, resume equivalence).

Full experiments ran on a Colab T4 (about [FILL] minutes per run):

1. Open `PathMNIST_FedAvg_vs_FedProx_Colab.ipynb` in Google Colab and set the runtime to GPU.
2. Add `WANDB_API_KEY` in Colab Secrets (key from https://wandb.ai/authorize). Never commit the key.
3. Upload `colab_package.zip`, mount Google Drive, and run the cells in order (data generation, checkpoint verification, tests, pilot, full orchestrator).
4. Download the results folder from Drive and run `python collect_results.py --results_dir <path>` to populate `submission/`.

Seeds: 42, 123, 2026. Partitions are generated once per seed and saved (hashes in `data/partitions/manifest.json`). The same code path produced all reported runs.

## 13. Repository layout

```
dataset / data generation:  generate_pathmnist_data.py, generate_initial_models.py
training:                   federated_runner.py, orchestrator.py, metrics_evaluator.py
reporting:                  generate_tables.py, generate_plots.py, generate_report_pdf.py, collect_results.py
tests/                      11 correctness tests
checkpoints/initial/        per-seed initial models + manifest.json
submission/                 plots/, metrics/, summary.pdf
PFLlib/                     upstream library (Apache 2.0), model architecture used
DEVIATIONS.md               audited differences from PFLlib and environment notes
```
