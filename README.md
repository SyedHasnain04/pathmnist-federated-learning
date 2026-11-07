# PathMNIST Federated Learning: FedAvg vs FedProx Under Non-IID Label Noise

## 1. Experimental Objective & Design

This repository contains the rigorous implementation, benchmarking, and empirical analysis of **FedAvg** versus **FedProx** under non-IID data partitioning with controlled label noise on the **PathMNIST** (MedMNIST v2) dataset.

### 2×2 Factorial Design
The experiment evaluates four distinct experimental groups across 3 independent random seeds (**42**, **123**, **2026**), yielding **12 final runs**:

| Algorithm | Condition | Description | Seeds |
|---|---|---|---|
| **FedAvg** | **Clean** | Standard Federated Averaging on clean non-IID client partitions | 42, 123, 2026 |
| **FedAvg** | **Noisy** | FedAvg with 40% symmetric label noise on Client 0 only | 42, 123, 2026 |
| **FedProx** | **Clean** | Federated Proximal with frozen proximal term on clean partitions | 42, 123, 2026 |
| **FedProx** | **Noisy** | FedProx with frozen proximal term and 40% noise on Client 0 | 42, 123, 2026 |

In addition, a pre-experimental **μ sweep** on noisy seed 42 evaluates candidates \(\mu \in \{0.001, 0.01, 0.1\}\) to freeze the optimal proximal parameter for all FedProx runs.

---

## 2. Dataset Provenance & Official Split Preservation

- **Dataset:** PathMNIST (colon pathology histology images, 9 classes, RGB 28×28).
- **Official Splits Preserved:**
  - Official Training: 89,996 images (MD5 verified against official MedMNIST release).
  - Official Validation: 10,004 images (used strictly for in-distribution evaluation and μ selection).
  - Official Test: 7,180 images (used strictly for final cross-site clinical evaluation).
- **Split Integrity Audit:** PFLlib's default dataset generator merges train and test splits into a single array before partitioning. To guarantee zero data leakage, our pipeline completely bypasses PFLlib's generator. Client partitions are generated exclusively from the official 89,996 training samples. Official validation and test sets remain pristine, read-only evaluation sets.

---

## 3. Client Partitioning & Noise Injection

1. **Non-IID Partitioning:**
   - 5 federated clients (Client 0 through Client 4).
   - Dirichlet distribution \(\mathrm{Dir}(\alpha=0.3)\) per class over official training images.
   - Deterministic client assignment with pre-split sample minimum \(\ge 1,000\) per client verified across all seeds.
   - Deterministic 75:25 stratified local split: 75% for client training, 25% for client-local in-distribution holdout evaluation.
2. **Label Noise Injection:**
   - **Client 0 is strictly designated as the noisy client.** Clients 1–4 receive 0% noise.
   - Exactly 40% symmetric label noise applied exclusively to Client 0's local training labels.
   - Noise sample count: \(n_{\mathrm{noisy}} = \mathrm{round}(0.4 \times n_0)\) using Python's round (half-to-even).
   - Replacement labels sampled uniformly at random from \(\{0, \dots, 8\} \setminus \{y\}\) without self-replacement.
   - Clean and noisy partition artifacts are saved independently.

### Client Sample Counts & Aggregation Weights

| Seed | Client 0 (Train / Noisy) | Client 1 (Train) | Client 2 (Train) | Client 3 (Train) | Client 4 (Train) | C0 Agg Weight (\(p_0\)) |
|---|---|---|---|---|---|---|
| **42** | 7,910 (3,164 noisy = 40.0%) | 15,460 | 12,981 | 16,113 | 15,032 | **0.1172** |
| **123** | 23,101 (9,240 noisy = 40.0%) | 15,060 | 2,297 | 14,355 | 12,683 | **0.3423** |
| **2026** | 8,874 (3,550 noisy = 40.0%) | 6,415 | 20,474 | 14,643 | 17,090 | **0.1315** |

> **Important Note on Client 0 Aggregation Weight Variation Across Seeds:**
> Due to Dirichlet non-IID partitioning (\(\alpha=0.3\)), Client 0's aggregation weight (\(p_0\)) differs substantially across seeds:
> - **Seed 42:** \(p_0 = 0.1172\) (7,910 samples, 3,164 noisy)
> - **Seed 123:** \(p_0 = 0.3423\) (23,101 samples, 9,240 noisy)
> - **Seed 2026:** \(p_0 = 0.1315\) (8,874 samples, 3,550 noisy)
> Because Client 0's local training labels contain 40% symmetric noise, Seed 123 exposes the global model to nearly 3× the noise mass compared to Seeds 42 and 2026. Consequently, experimental results must be interpreted both per-seed and as mean ± std across seeds; reporting only aggregate means masks this structural sensitivity. Per-seed values are presented alongside mean ± std in Table 2 (`results/tables/results.md`) and plotted as individual seed curves in `results/plots/plot1_test_accuracy_vs_rounds.png`.

---

## 4. Model Architecture & Initialization Checkpoints

- **Model:** `FedAvgCNN` from PFLlib (`in_features=3, num_classes=9, dim=1024`).
- **Layers:** 2 Convolutional layers (16 and 32 filters, 5×5 kernel, padding=2), ReLU, 2×2 MaxPool, followed by fully connected layers (1024 → 512 → 9).
- **Parameter Count:** 583,113 trainable parameters. No BatchNorm layers (eliminating non-trainable buffer synchronization discrepancies).
- **Deterministic Checkpoints:** Identical initial checkpoints (`checkpoints/initial/initial_model_seed_{seed}.pt`) are generated deterministically per seed and loaded by all 4 experimental groups within that seed.

---

## 5. Optimization & Algorithmic Equivalence

- **Rounds:** 100 global communication rounds.
- **Local Epochs:** \(E = 5\) local epochs per client per round.
- **Batch Size:** 64.
- **Optimizer:** Standard PyTorch `torch.optim.SGD(lr=0.01, momentum=0.9, weight_decay=0.0)`.
- **Optimizer Lifecycle:** Fresh optimizer instance created at the beginning of each client/round training invocation. **Zero momentum carried over across communication rounds.**
- **FedProx Formulation:**
  \[
  \mathcal{L}_{\mathrm{FedProx}}(w) = \mathcal{L}_{\mathrm{CE}}(w) + \frac{\mu}{2} \sum_{l} \|w_l - w_{\mathrm{global}}^t\|_2^2
  \]
  The proximal term is anchored to a detached snapshot \(w_{\mathrm{global}}^t\) taken at the start of the round.
- **Audit of PFLlib:** PFLlib's `PerturbedGradientDescent` optimizer omits momentum entirely. To preserve the specification's strict SGD momentum (0.9) requirement, PyTorch SGD is used with explicit proximal loss penalty.
- **Deterministic Equivalence Test:** Unit test verifies that with \(\mu=0.0\), FedProx is numerically identical to FedAvg (\(\max |\Delta w| < 10^{-7}\)).

---

## 6. Aggregation & Update Metrics

### Aggregation Formula
Sample-count-weighted aggregation with full participation (\(\mathrm{join\_ratio}=1.0\)):
\[
w_{\mathrm{global}}^{t+1} = \sum_{k=0}^{4} p_k w_{k,\mathrm{local}}^t, \quad p_k = \frac{n_k}{\sum_j n_j}
\]

### Diagnostic Update Metrics (Pre-Aggregation)
1. **Per-client update:** \(\Delta_k^t = w_{k,\mathrm{local}}^t - w_{\mathrm{global}}^t\).
2. **Update L2 norm:** \(\|\Delta_k^t\|_2\).
3. **Aggregated global update:** \(\Delta_{\mathrm{global}}^t = \sum_k p_k \Delta_k^t\).
4. **Standard Cosine Similarity:** \(\cos(\Delta_k^t, \Delta_{\mathrm{global}}^t)\).
5. **Leave-One-Out Cosine Similarity:** \(\cos(\Delta_k^t, \Delta_{-k}^t)\), where \(\Delta_{-k}^t = \frac{\sum_{j \neq k} n_j \Delta_j^t}{\sum_{j \neq k} n_j}\).

---

## 7. Evaluation & Metric Isolation

| Evaluation Type | Split Source | Samples | Purpose |
|---|---|---|---|
| **Official Validation** | `data/partitions/official/val.npz` | 10,004 | In-distribution tracking & μ selection |
| **Official Global Test** | `data/partitions/official/test.npz` | 7,180 | Final cross-site clinical generalization |
| **Client Local Holdout** | `data/partitions/seed_{seed}/{cond}/test/{cid}.npz` | ~2,000–7,000 | In-distribution client holdout performance |

---

## 8. μ Sweep Specification & Selection

- **Candidates:** \(\mu \in \{0.001, 0.01, 0.1\}\).
- **Condition:** Noisy setting, seed 42, 100 rounds.
- **Selection Rule:** Highest mean official validation accuracy across rounds 91–100. Ties broken in favor of smaller μ.
- **Freezing:** Selected μ frozen for all FedProx runs (clean and noisy) across seeds 42, 123, 2026.
- **Reuse:** Winning sweep run reused directly as `fedprox_noisy_seed42`.

---

## 9. Deliverables & Artifacts

- `PathMNIST_FedAvg_vs_FedProx_Report.pdf`: One-page executive summary PDF report.
- `results/tables/results.md` & `all_results.json`: Full numerical result tables.
- `results/plots/`: High-resolution figures for all required diagnostic and evaluation curves.
- `DEVIATIONS.md`: Complete record of environmental adaptations and audited PFLlib behaviors.
- `tests/`: 8 comprehensive correctness tests passing.

---

## 10. How to Run & Reproduce

```bash
# 1. Run all unit and correctness tests
python -m pytest tests/ -v

# 2. Run the automated experiment orchestrator (executes sweep, μ selection, and all 12 runs)
python orchestrator.py

# 3. Generate tables, plots, and PDF report from saved JSON logs
python generate_tables.py
python generate_plots.py
python generate_report_pdf.py
```

---

## 11. PFLlib Modifications, Licensing & Attribution

### Attribution & Upstream License
This project utilizes the `FedAvgCNN` backbone model architecture from **PFLlib** (Personalized Federated Learning Library) developed by Jianxu Ji et al.
PFLlib is licensed under the **Apache License, Version 2.0**. In accordance with Section 4 of the Apache License 2.0, the complete, unmodified original license file is preserved at [`PFLlib/LICENSE`](file:///c:/Users/Syedh/Downloads/Research/llmXfedrated/PFLlib/LICENSE).

### Comprehensive Audit of PFLlib Modifications
To adhere strictly to the authoritative experimental specification and prevent experimental invalidity, the following modifications and architectural decouplings were implemented:

1. **Decoupled Data Pipeline (Eliminated Train/Test Leakage):**
   - *Upstream PFLlib behavior:* Dataset generators in PFLlib (e.g., `generate_MNIST.py`) merge official training and testing splits into a single unified array prior to Dirichlet partitioning.
   - *Modification:* PFLlib's data generation logic is completely bypassed. Our data generator (`generate_pathmnist_data.py`) partitions solely from the 89,996 official PathMNIST training images. Official in-distribution validation (10,004) and official cross-site test (7,180) splits remain completely detached and untouched.

2. **Momentum-Preserving Optimizer State Lifecycle:**
   - *Upstream PFLlib behavior:* PFLlib's `PerturbedGradientDescent` optimizer implementation omits momentum entirely.
   - *Modification:* Replaced with standard PyTorch `torch.optim.SGD(lr=0.01, momentum=0.9)`. A fresh optimizer instance is created for each client at the beginning of each round, ensuring momentum buffers from round $t$ never bleed into round $t+1$.

3. **Detached Snapshot Anchoring for FedProx:**
   - *Upstream PFLlib behavior:* FedProx implementations in PFLlib can exhibit parameter reference sharing issues across local steps.
   - *Modification:* In `federated_runner.py`, a detached, cloned copy of global parameters $w_{\mathrm{global}}^t$ is captured at round start. The proximal objective $\frac{\mu}{2} \sum \|w - w_{\mathrm{global}}^t\|_2^2$ is penalized directly in the backward pass.

4. **Pre-Aggregation Diagnostic Metrics:**
   - *Upstream PFLlib behavior:* PFLlib records basic training and test accuracy without gradient alignment diagnostics.
   - *Modification:* Captured local parameter vectors prior to aggregation to compute per-client update L2 norms, standard cosine similarities, and leave-one-out cosine similarities against peer client updates.

5. **Deterministic Checkpoint Harmonization:**
   - *Upstream PFLlib behavior:* Initializes models randomly upon client construction.
   - *Modification:* Implemented deterministic initial checkpoint generation per seed with SHA-256 manifest verification, guaranteeing all 4 experimental groups within a seed begin from identical parameters.

