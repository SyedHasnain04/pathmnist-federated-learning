# Final Execution Plan: FedAvg vs FedProx on PathMNIST (PFLlib)

**FROZEN EXPERIMENTAL DESIGN — DO NOT ADD VARIABLES**

Purpose: authoritative implementation specification for the PFLlib audit, leakage-safe PathMNIST pipeline, FedAvg/FedProx implementation, experiments, reporting, and submission.

**Execution rule:** Do not silently change the design. If a blocker arises, apply the smallest technically sound fallback, document it in `DEVIATIONS.md`, and continue. Stop only if official train/test separation cannot be guaranteed. Never fabricate results or claim incomplete work is complete.

## 1. Fixed configuration

| Parameter | Frozen value |
|---|---|
| Dataset | PathMNIST (MedMNIST), RGB 3×28×28, 9 classes |
| Clients | 5; `join_ratio = 1.0` |
| Partition | Class-wise Dirichlet, α = 0.3; official training split only |
| Minimum client size | At least 1,000 assigned samples before local split; reject/regenerate partition if any client is below threshold |
| Client train/test split | 75:25, seeded, non-stratified; save exact indices |
| Model | PFLlib FedAvgCNN; first convolution has 3 input channels; expected flattened FC input 1024 for the standard architecture, confirmed with a dummy `(1, 3, 28, 28)` forward pass |
| Rounds / local epochs / batch | 100 / 5 / 64 |
| Optimizer | SGD, `lr=0.01`, `momentum=0.9`, `weight_decay=0`; fresh optimizer every communication round |
| Aggregation | Sample-count weighted using actual local training sample counts |
| FedProx μ | Selected by sweep in Section 6 |
| Seeds | 42, 123, 2026 |
| Noise | 40% of Client 0 local training labels; exact count is Python `round(0.4 * n0)` |

## 2. Experiment matrix — 12 final runs

| Algorithm | Clean | 40% noise on Client 0 |
|---|---|---|
| FedAvg | `fedavg_clean_seed42`, `fedavg_clean_seed123`, `fedavg_clean_seed2026` | `fedavg_noisy_seed42`, `fedavg_noisy_seed123`, `fedavg_noisy_seed2026` |
| FedProx | `fedprox_clean_seed42`, `fedprox_clean_seed123`, `fedprox_clean_seed2026` | `fedprox_noisy_seed42`, `fedprox_noisy_seed123`, `fedprox_noisy_seed2026` |

The 2×2 design supports algorithm-effect, noise-effect, and interaction comparisons. With three seeds, do not claim statistical significance.

The μ sweep adds three candidate jobs unless the selected sweep run can be safely reused as `fedprox_noisy_seed42`. Reuse only if configuration, code commit, artifact hashes, deterministic settings, and all 100 rounds match the final specification; otherwise rerun.

## 3. Data pipeline — verify before training

1. Audit PFLlib's dataset-generation code, especially `generate_MNIST.py`. Determine whether its pattern merges official train and test data. Do not copy any train/test merge behavior for PathMNIST.
2. Preserve official splits: official train is used only for client partitioning; official validation only for validation and μ selection; official test only for global cross-site evaluation.
3. For each seed, partition only official training data using class-wise Dirichlet allocation (α = 0.3), then split each client's assigned partition 75:25 using a seeded, non-stratified split.
4. Save partition indices and client-local train/test indices as immutable artifacts. All four experiment groups for a seed load exactly the same artifacts. Never regenerate at run time.
5. Enforce at least 1,000 assigned official-training samples per client before the 75:25 split. Reject/regenerate if the constraint fails; document this as a constrained Dirichlet partition.
6. Check every client has non-empty local train and test sets. Record counts and class distributions.
7. Keep official validation and official test completely outside client partitions and client-local test sets. If this separation cannot be guaranteed, stop.

## 4. Label noise

- Apply symmetric label noise to Client 0's local training labels only.
- Use exactly Python's built-in `round(0.4 * n0)`, including ties-to-even behavior, for both generation and assertions.
- Select corrupted indices uniformly without replacement. Choose replacement labels uniformly from the other eight classes; a replacement must never equal the original.
- Apply corruption when generating saved artifacts, not dynamically during loading. Save separate clean and noisy `.npz` files. Clean groups use original labels at the same selected indices.
- Images and sample order must be identical between clean/noisy artifacts; only intended Client 0 training labels may differ.
- Save noise indices, original/replacement labels, counts, and hashes.
- Assert exact noise count, no unchanged replacement labels, no overlap with Client 0 local test indices, and unchanged client-test, official-validation, official-test, and clean-artifact hashes.

## 5. Initialization and reproducibility

- For each seed, set Python, NumPy, PyTorch CPU, and CUDA seeds before model initialization.
- Instantiate one initial model per seed, save its checkpoint, and compute a canonical SHA256 hash.
- All four groups for a seed load the exact same initial checkpoint. Assert parameter equality and matching hashes.
- Record seed, initial model hash, partition/split/noise artifact hashes, software versions, configuration, and Git commit in W&B and the run manifest.
- If deterministic CUDA settings are used for sweep-run reuse, record and consistently apply them.

## 6. FedProx implementation and μ sweep

FedProx objective:

\[
F_k(w) + \frac{\mu}{2}\lVert w-w_t\rVert^2
\]

Here \(w_t\) is a detached snapshot of the global trainable parameters captured at the beginning of round \(t\). It remains fixed throughout local training.

- Audit PFLlib `clientprox`/`serverprox`. Confirm the proximal term contributes to the gradient and references the fixed round-start global parameters.
- Both FedAvg and FedProx must use the same `torch.optim.SGD(lr=0.01, momentum=0.9, weight_decay=0)` implementation.
- Create a fresh optimizer at the start of each client/round local-training call. Momentum state must not carry between communication rounds, but momentum operates normally within local training.
- Do not assume PFLlib's stock `PerturbedGradientDescent` has momentum. If it is plain SGD or mismatched, adapt FedProx to use the same PyTorch SGD optimizer as FedAvg.
- Implement the proximal gradient either by adding `μ * (p - g_round_start)` to each trainable parameter's gradient before `optimizer.step()`, or by adding the equivalent `(μ/2) * sum(||p - g_round_start||²)` term to the loss.
- Add a deterministic equivalence test: with `μ=0`, identical initialization, data order, batches, seeds, optimizer, and participation, FedProx must match FedAvg exactly or within documented floating-point tolerance.
- Capture local parameters before aggregation so update metrics use actual client updates.

**μ sweep — fixed in advance:**

1. Candidates: \(\mu \in \{0.001, 0.01, 0.1\}\).
2. Run only on noisy setting, seed 42.
3. Select highest mean official validation accuracy over rounds 91–100; ties go to smaller μ.
4. Freeze selected μ for every FedProx clean/noisy final run and all three seeds.
5. Disclose that the sweep uses one seed and in-distribution validation; it is preliminary and may not optimize cross-site test performance.

**Momentum limitation:** The proximal gradient contribution is accumulated through SGD momentum, so the practical effect of μ depends on optimizer dynamics and is not directly comparable to interpreting μ independently of momentum.

## 7. Evaluation

| Evaluation | Data source | Frequency / interpretation |
|---|---|---|
| Global test accuracy/loss | Official PathMNIST test | Every round; cross-site |
| Validation accuracy/loss | Official PathMNIST validation | Every round; in-distribution model selection |
| Per-client accuracy/loss | Each client's held-out 25% of official-training partition | Every round; in-distribution client holdout |
| Client test sample counts | Each client-local test split | Per client, per run |
| Confusion matrix | Official PathMNIST test | Final round only |

- Audit PFLlib `evaluate()`; it is expected to use client-local split files. Use it only for client-local evaluation after verifying behavior.
- Implement separate custom official-validation and official-global-test evaluators. Unit-test them with a trivial/deterministic model and small known inputs.
- Verify shapes, class dimension, dtypes, label mapping, sample counts, loss, accuracy, source split, and that evaluation does not mutate model parameters.
- Use plot titles that explicitly label official test as cross-site and client/validation metrics as in-distribution.
- Never average client-local metrics and official global-test metrics or present them as equivalent distributions.

## 8. Update norms and cosine similarity

Flatten trainable parameters only. Exclude non-trainable buffers and BatchNorm running statistics. The intended FedAvgCNN has no BatchNorm; verify the actual model.

\[
\Delta_k^t = w_{k,\mathrm{local}}^t - w_{\mathrm{global}}^t
\]

\[
\lVert\Delta_k^t\rVert_2
\]

\[
\Delta_{\mathrm{global}}^t =
\sum_k \frac{n_k}{N}\Delta_k^t
= w_{\mathrm{global}}^{t+1}-w_{\mathrm{global}}^t
\]

**Primary standard cosine:**

\[
\cos(\Delta_k^t,\Delta_{\mathrm{global}}^t)
\]

**Additional leave-one-out cosine:**

\[
\cos(\Delta_k^t,\Delta_{-k}^t),\qquad
\Delta_{-k}^t =
\frac{\sum_{j\ne k} n_j\Delta_j^t}{\sum_{j\ne k}n_j}
\]

**Aggregation weight:**

\[
p_k=\frac{n_k}{\sum_j n_j}
\]

Use actual local training sample counts used by PFLlib.

- Capture each client's local parameters before aggregation. Compute the weighted global update from those same client updates.
- Compute standard cosine (primary) and leave-one-out cosine (additional diagnostic). Standard cosine includes the client's own contribution and can inflate apparent alignment.
- If either vector norm is zero, log cosine as undefined/NaN, not zero.
- Log each client's update norm, both cosine values, aggregation weight, and training sample count every round.
- Plot clean and noisy conditions side by side. Low cosine can arise from non-IID data alone; compare against clean controls before attributing misalignment to label noise.

## 9. Weights & Biases

Project: `pathmnist-federated-learning` (public only after public visibility is verified). Create 12 final runs with the names in Section 2.

Run config: dataset, clients, α, noise rate/client ID, learning rate, momentum, local epochs, batch size, seed, algorithm, μ, architecture, initial model hash, artifact hashes, package versions, and Git commit.

**Per round, log:**
- Official global test accuracy/loss and official validation accuracy/loss.
- Per-client local test accuracy/loss and client-local test sample counts.
- Per-client update norms, standard cosine, leave-one-out cosine, training sample counts, aggregation weights, and participation.
- Client 0 assigned/train/test sizes, class distribution, noise count/rate, and aggregation weight.

Never commit W&B API keys or credentials. If credentials/network are unavailable, implement local/offline logging and do not claim a public W&B project exists.

## 10. Reporting

Primary result: round 100. Robustness summary: mean over rounds 91–100. Report mean ± standard deviation across three seeds and individual-seed curves. Do not claim statistical significance.

**Required plots:**
1. Global official-test (cross-site) accuracy versus rounds.
2. Global official-test (cross-site) loss versus rounds.
3. Per-client in-distribution accuracy.
4. Official validation (in-distribution) accuracy.
5. Per-client update norms.
6. Per-client standard cosine similarity.
7. Per-client leave-one-out cosine similarity.
8. Client class-distribution heatmap (5×9).
9. Client aggregation weights, clearly identifying Client 0.

Interpret clean versus noisy and FedAvg versus FedProx. Discuss whether Client 0 stands out in norm/cosine, its aggregation weight, whether FedProx changes the signal, and whether non-IID heterogeneity explains observations. Do not assume FedProx performs better; null/negative results are valid.

## 11. Execution order and deadline checkpoint

**Deadline: 10 November 2026, 11:59 PM.** Confirm if the organizer's official deadline differs.

1. Clone PFLlib and audit dataset generation, split seeding, FedAvgCNN, FedProx, evaluation, and aggregation.
2. Confirm GPU availability, model/memory, CUDA and PyTorch compatibility. Time one end-to-end round including training, aggregation, evaluation, metrics, and logging. Estimate sweep/final-run time before launching.
3. Write PathMNIST generator preserving official splits.
4. Verify FedAvgCNN RGB 28×28 forward pass and expected 1024 FC input for the standard architecture.
5. Generate partitions for all three seeds; check 1,000-sample pre-split minimum, counts, Client 0 size/class distribution, and aggregation weight.
6. Generate local splits and verify saved indices/hashes.
7. Test clean/noisy artifacts and all noise assertions.
8. Verify identical initial checkpoints across four groups per seed.
9. Verify FedProx proximal reference, gradient contribution, optimizer parity/lifecycle, and μ=0 equivalence.
10. Unit-test official validation/test evaluators on a trivial model.
11. Run μ sweep on noisy seed 42 and freeze μ using mean validation accuracy over rounds 91–100.
12. Run approximately five pilot rounds for all four groups to verify data loading, evaluation, W&B, and update metrics.
13. Run all 12 final configurations. Reuse sweep run only if all strict reproducibility conditions are met.
14. Generate plots, tables, README, repository, one-page experiment PDF, separate Reactive Optimizers summary if required, and submission email.

**Priority cut line:**
- Essential first: official split integrity; reproducible partition/noise pipeline; correct FedAvg/FedProx optimizer parity and μ=0 test; GPU timing pilot; μ sweep; 12 final runs; W&B or complete local fallback logs; accurate README; required PDF(s).
- Secondary after essentials: extra test-file coverage beyond critical correctness tests, confusion matrices, and visual polish.
- Never fabricate metrics or imply unfinished runs completed. If full runs cannot finish, report exactly what ran, failed, and remains.

## 12. Limitations to disclose

1. Official PathMNIST test represents a different clinical-site distribution; cross-site performance may be lower than in-distribution performance. This is an evaluation distinction, not automatically a bug.
2. Validation is in-distribution, so μ selection may not optimize cross-site generalization.
3. Per-client test sets may be small and non-IID; accuracy can be noisy. Always report sample counts.
4. If Client 0 has a small aggregation weight, its label noise may have limited global influence; a null result is possible and valid.
5. Three seeds provide weak variability estimates; do not make statistical-significance claims.
6. Momentum 0.9 changes the practical effect of the proximal term; μ is not directly comparable to results using different optimizer dynamics.
7. The μ sweep uses one seed only and is preliminary.

## 13. Deliverables

- GitHub repository with code, configs, partition procedure, tests, results, plots, and README documenting every PFLlib modification and train/test leakage audit.
- Public W&B project with 12 final runs, per-round metrics, and comparison dashboard, if credentials/network allow and public visibility is verified.
- One-page experiment PDF: objective, setup, results, plots, update-metric observations, conclusion, and limitations.
- Email containing GitHub URL, W&B URL (if genuinely public), and PDF.
- Confirm with organizers whether the one-page Reactive Optimizers post summary is a separate Step 2 deliverable. Do not delay technical implementation while awaiting clarification.

## 14. Immediate next steps

- [ ] Audit PFLlib dataset generator for train/test merging.
- [ ] Verify client split seeding and disable hidden participation/training variations.
- [ ] Inspect FedAvgCNN and FedProx implementations, including optimizer behavior.
- [ ] Confirm GPU availability and time one complete communication round.
- [ ] Create PathMNIST generator preserving official splits.
- [ ] Generate and inspect partitions for all three seeds; verify Client 0 sample count and aggregation weight.

## 15. PFLlib settings that must not silently alter the design

Explicitly inspect and disable or neutralize:
- `learning_rate_decay`
- `train_slow`
- `client_drop_rate`
- `random_join_ratio`
- time-threshold filtering

Confirm `join_ratio=1.0` results in all five clients participating every round. Log effective settings in each run.

## 16. README minimum content

- Objective and 2×2 experiment matrix.
- Dataset provenance and official split preservation; explicit statement that official test is never merged into client data.
- Model architecture and verified input/output dimensions.
- Frozen hyperparameters; Dirichlet partition and minimum-size constraint; non-stratified local split.
- Client 0 noise definition, artifact generation, hashes, and assertions.
- Seeds, initial checkpoints, and reproducibility procedure.
- FedAvg/FedProx implementation, proximal reference, optimizer parity, and μ=0 equivalence test.
- Sample-count aggregation formula and client weights.
- Evaluation distribution distinctions: client holdout, official validation, official cross-site test.
- Update norm, standard cosine, and leave-one-out cosine definitions.
- W&B project/runs or offline logging status.
- Commands for data generation, μ sweep, experiments, tests, and plots.
- Environment/dependency versions, Git commit, actual results only, mean ± std, round-100 and last-10-round summaries.
- Limitations, troubleshooting, and verified PFLlib/MedMNIST citations.
