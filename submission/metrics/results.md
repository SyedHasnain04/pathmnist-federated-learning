# PathMNIST FedAvg vs FedProx — Experiment Results

## Table 1: Per-Run Results at Round 100

| Run Name | Algorithm | Condition | Seed | Rounds | Test@100 | Val@100 | Test Mean 91-100 | Val Mean 91-100 | C0 Holdout@100 |
|----------|-----------|-----------|------|--------|----------|---------|-----------------|-----------------|----------------|
| fedavg_clean_seed42 | FEDAVG | clean | 42 | 100/100 | 0.7777 | 0.8516 | 0.7749 | 0.8510 | 0.8297 |
| fedavg_clean_seed123 | FEDAVG | clean | 123 | 100/100 | 0.7577 | 0.8474 | 0.7629 | 0.8488 | 0.9036 |
| fedavg_clean_seed2026 | FEDAVG | clean | 2026 | 100/100 | 0.7901 | 0.8465 | 0.7768 | 0.8467 | 0.9017 |
| fedavg_noisy_seed42 | FEDAVG | noisy | 42 | 100/100 | 0.7532 | 0.8188 | 0.7491 | 0.8180 | 0.7539 |
| fedavg_noisy_seed123 | FEDAVG | noisy | 123 | 100/100 | 0.6455 | 0.6748 | 0.6500 | 0.6740 | 0.6717 |
| fedavg_noisy_seed2026 | FEDAVG | noisy | 2026 | 100/100 | 0.7506 | 0.7958 | 0.7457 | 0.7945 | 0.8287 |
| fedprox_clean_seed42 | FEDPROX | clean | 42 | 100/100 | 0.7659 | 0.8650 | 0.7706 | 0.8645 | 0.8578 |
| fedprox_clean_seed123 | FEDPROX | clean | 123 | 100/100 | 0.7756 | 0.8377 | 0.7645 | 0.8433 | 0.9146 |
| fedprox_clean_seed2026 | FEDPROX | clean | 2026 | 100/100 | 0.7638 | 0.8645 | 0.7667 | 0.8631 | 0.8824 |
| fedprox_noisy_seed42 | FEDPROX | noisy | 42 | 100/100 | 0.7666 | 0.8464 | 0.7538 | 0.8455 | 0.8305 |
| fedprox_noisy_seed123 | FEDPROX | noisy | 123 | 100/100 | 0.6831 | 0.7967 | 0.6880 | 0.7920 | 0.8122 |
| fedprox_noisy_seed2026 | FEDPROX | noisy | 2026 | 100/100 | 0.7323 | 0.8404 | 0.7289 | 0.8370 | 0.8442 |

## Table 2: Group Summary (Per-Seed Values and Mean ± Std across 3 seeds)

*Note: Client 0 aggregation weight (p0) differs by seed: 0.1172 (seed 42), 0.3423 (seed 123), 0.1315 (seed 2026).*

| Group | Seed 42 (p0=0.117) | Seed 123 (p0=0.342) | Seed 2026 (p0=0.132) | Test Mean 91-100 (Mean ± Std) | Val Mean 91-100 | C0 Holdout@100 |
|-------|--------------------|---------------------|----------------------|--------------------------------|-----------------|----------------|
| FEDAVG clean | 0.7749 | 0.7629 | 0.7768 | 0.7715 ± 0.0061 | 0.8488 ± 0.0017 | 0.8783 ± 0.0344 |
| FEDAVG noisy | 0.7491 | 0.6500 | 0.7457 | 0.7149 ± 0.0459 | 0.7622 ± 0.0631 | 0.7514 ± 0.0641 |
| FEDPROX clean | 0.7706 | 0.7645 | 0.7667 | 0.7673 ± 0.0025 | 0.8570 ± 0.0097 | 0.8849 ± 0.0232 |
| FEDPROX noisy | 0.7538 | 0.6880 | 0.7289 | 0.7236 ± 0.0271 | 0.8248 ± 0.0234 | 0.8290 ± 0.0131 |

## Table 3: mu Sweep Results (FedProx, noisy, seed=42)

**Selected mu = 0.1** (criterion: highest mean official val accuracy rounds 91-100; ties broken by smallest mu)

| mu | Rounds | Val Mean 91-100 | Test Mean 91-100 | Val@100 | Selected? |
|---|--------|-----------------|-----------------|---------|-----------|
| 0.001 | 100/100 | 0.8170 | 0.7383 | 0.8172 |  |
| 0.01 | 100/100 | 0.8275 | 0.7456 | 0.8283 |  |
| 0.1 | 100/100 | 0.8455 | 0.7538 | 0.8464 | YES |