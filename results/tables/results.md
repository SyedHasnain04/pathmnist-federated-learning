# PathMNIST FedAvg vs FedProx — Experiment Results

## Table 1: Per-Run Results at Round 100

| Run Name | Algorithm | Condition | Seed | Rounds | Test@100 | Val@100 | Test Mean 91-100 | Val Mean 91-100 | C0 Holdout@100 |
|----------|-----------|-----------|------|--------|----------|---------|-----------------|-----------------|----------------|
| fedavg_clean_seed42 | FEDAVG | clean | 42 | 0/100 | N/A | N/A | N/A | N/A | N/A |
| fedavg_clean_seed123 | FEDAVG | clean | 123 | 0/100 | N/A | N/A | N/A | N/A | N/A |
| fedavg_clean_seed2026 | FEDAVG | clean | 2026 | 0/100 | N/A | N/A | N/A | N/A | N/A |
| fedavg_noisy_seed42 | FEDAVG | noisy | 42 | 0/100 | N/A | N/A | N/A | N/A | N/A |
| fedavg_noisy_seed123 | FEDAVG | noisy | 123 | 0/100 | N/A | N/A | N/A | N/A | N/A |
| fedavg_noisy_seed2026 | FEDAVG | noisy | 2026 | 0/100 | N/A | N/A | N/A | N/A | N/A |
| fedprox_clean_seed42 | FEDPROX | clean | 42 | 0/100 | N/A | N/A | N/A | N/A | N/A |
| fedprox_clean_seed123 | FEDPROX | clean | 123 | 0/100 | N/A | N/A | N/A | N/A | N/A |
| fedprox_clean_seed2026 | FEDPROX | clean | 2026 | 0/100 | N/A | N/A | N/A | N/A | N/A |
| fedprox_noisy_seed42 | FEDPROX | noisy | 42 | 0/100 | N/A | N/A | N/A | N/A | N/A |
| fedprox_noisy_seed123 | FEDPROX | noisy | 123 | 0/100 | N/A | N/A | N/A | N/A | N/A |
| fedprox_noisy_seed2026 | FEDPROX | noisy | 2026 | 0/100 | N/A | N/A | N/A | N/A | N/A |

## Table 2: Group Summary (Mean ± Std across 3 seeds)

| Group | N Seeds | Test@100 | Val@100 | Test Mean 91-100 | Val Mean 91-100 | C0 Holdout@100 |
|-------|---------|----------|---------|-----------------|-----------------|----------------|
| FEDAVG clean | 3 | N/A | N/A | N/A | N/A | N/A |
| FEDAVG noisy | 3 | N/A | N/A | N/A | N/A | N/A |
| FEDPROX clean | 3 | N/A | N/A | N/A | N/A | N/A |
| FEDPROX noisy | 3 | N/A | N/A | N/A | N/A | N/A |

## Table 3: mu Sweep Results (FedProx, noisy, seed=42)

**Selected mu = not_selected** (criterion: highest mean official val accuracy rounds 91-100; ties broken by smallest mu)

| mu | Rounds | Val Mean 91-100 | Test Mean 91-100 | Val@100 | Selected? |
|---|--------|-----------------|-----------------|---------|-----------|
| 0.001 | 0/100 | N/A | N/A | N/A |  |
| 0.01 | 0/100 | N/A | N/A | N/A |  |
| 0.1 | 0/100 | N/A | N/A | N/A |  |