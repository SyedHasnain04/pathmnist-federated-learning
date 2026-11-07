"""
launch_experiments.py - Orchestrates parallel federated learning experiments.

Usage:
    python launch_experiments.py --phase mu_sweep    # Launch 3 mu sweep runs
    python launch_experiments.py --phase fedavg      # Launch 6 FedAvg runs
    python launch_experiments.py --phase fedprox --mu 0.01  # Launch 6 FedProx runs
    python launch_experiments.py --phase all --mu 0.01      # Launch all 12 final runs

Runs are launched as background subprocesses. Progress is monitored via log files.
"""
import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
THREADS_PER_RUN = 4  # 4 threads each; run 4-5 in parallel = 16-20 cores
ROUNDS = 100

# Experiment matrix
SEEDS = [42, 123, 2026]

MU_SWEEP_CONFIGS = [
    {"run_name": "fedprox_sweep_mu0001", "algorithm": "fedprox", "condition": "noisy", "seed": 42, "mu": 0.001},
    {"run_name": "fedprox_sweep_mu001",  "algorithm": "fedprox", "condition": "noisy", "seed": 42, "mu": 0.01},
    {"run_name": "fedprox_sweep_mu01",   "algorithm": "fedprox", "condition": "noisy", "seed": 42, "mu": 0.1},
]

FEDAVG_CONFIGS = [
    {"run_name": f"fedavg_clean_seed{s}", "algorithm": "fedavg", "condition": "clean", "seed": s, "mu": 0.0}
    for s in SEEDS
] + [
    {"run_name": f"fedavg_noisy_seed{s}", "algorithm": "fedavg", "condition": "noisy", "seed": s, "mu": 0.0}
    for s in SEEDS
]

def fedprox_configs(mu):
    return [
        {"run_name": f"fedprox_clean_seed{s}", "algorithm": "fedprox", "condition": "clean", "seed": s, "mu": mu}
        for s in SEEDS
    ] + [
        {"run_name": f"fedprox_noisy_seed{s}", "algorithm": "fedprox", "condition": "noisy", "seed": s, "mu": mu}
        for s in SEEDS
    ]

def run_is_complete(run_name):
    """Check if a run has a complete history.json with 100 rounds."""
    history_path = RESULTS_DIR / run_name / "history.json"
    if not history_path.exists():
        return False
    try:
        h = json.load(open(history_path))
        return len(h.get("rounds", [])) >= ROUNDS
    except Exception:
        return False

def launch_run(config, threads=THREADS_PER_RUN):
    """Launch a single experiment as a background subprocess."""
    run_name = config["run_name"]
    if run_is_complete(run_name):
        print(f"  [SKIP] {run_name} already complete.")
        return None
    
    cmd = [
        sys.executable, "-u", "federated_runner.py",
        "--run_name", run_name,
        "--algorithm", config["algorithm"],
        "--condition", config["condition"],
        "--seed", str(config["seed"]),
        "--mu", str(config.get("mu", 0.0)),
        "--rounds", str(ROUNDS),
        "--threads", str(threads),
    ]
    
    log_path = RESULTS_DIR / run_name / "stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=logf,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        )
    
    print(f"  [LAUNCHED] {run_name} | PID: {proc.pid} | Log: {log_path}")
    return proc

def wait_for_completion(procs_and_names, check_interval=60):
    """Wait until all processes are done."""
    while True:
        all_done = all(p.poll() is not None for p, _ in procs_and_names if p is not None)
        if all_done:
            break
        still_running = [(p, n) for p, n in procs_and_names if p is not None and p.poll() is None]
        print(f"  [{time.strftime('%H:%M:%S')}] Running: {[n for _, n in still_running]}")
        time.sleep(check_interval)
    print("  All processes completed.")

def select_mu_from_sweep():
    """Select best mu based on mean val acc rounds 91-100."""
    results = {}
    for cfg in MU_SWEEP_CONFIGS:
        run_name = cfg["run_name"]
        mu = cfg["mu"]
        history_path = RESULTS_DIR / run_name / "history.json"
        if not history_path.exists():
            print(f"  WARNING: {run_name} history.json not found!")
            continue
        h = json.load(open(history_path))
        rounds_data = h.get("rounds", [])
        # rounds 91-100 (1-indexed)
        late_rounds = [r for r in rounds_data if 91 <= r["round"] <= 100]
        if len(late_rounds) < 10:
            print(f"  WARNING: {run_name} has only {len(late_rounds)} rounds in 91-100 range.")
        val_accs = [r.get("official_val_accuracy", r.get("official_val_acc", 0)) for r in late_rounds]
        mean_val = sum(val_accs) / len(val_accs) if val_accs else 0
        results[mu] = {"run_name": run_name, "mean_val_91_100": mean_val, "n_rounds": len(late_rounds)}
        print(f"  mu={mu}: mean_val_acc_91_100 = {mean_val:.4f} (from {len(val_accs)} rounds)")
    
    if not results:
        raise ValueError("No sweep results found!")
    
    # Select mu with highest mean val acc; ties broken by smallest mu
    best_mu = max(results.keys(), key=lambda m: (results[m]["mean_val_91_100"], -m))
    print(f"\n  >>> Selected mu = {best_mu} (mean val acc = {results[best_mu]['mean_val_91_100']:.4f})")
    
    # Save selection
    selection = {"selected_mu": best_mu, "candidates": results, "selection_criterion": "max mean val acc rounds 91-100; ties broken by smallest mu"}
    with open(RESULTS_DIR / "mu_selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    print(f"  Saved to results/mu_selection.json")
    return best_mu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["mu_sweep", "fedavg", "fedprox", "all", "select_mu", "status"], default="status")
    parser.add_argument("--mu", type=float, default=None, help="FedProx mu (required for fedprox/all phases)")
    parser.add_argument("--threads", type=int, default=THREADS_PER_RUN)
    args = parser.parse_args()
    
    if args.phase == "status":
        all_configs = MU_SWEEP_CONFIGS + FEDAVG_CONFIGS
        if args.mu is not None:
            all_configs += fedprox_configs(args.mu)
        print("Run status:")
        for cfg in all_configs:
            complete = run_is_complete(cfg["run_name"])
            print(f"  {'[DONE]' if complete else '[----]'} {cfg['run_name']}")
    
    elif args.phase == "select_mu":
        select_mu_from_sweep()
    
    elif args.phase == "mu_sweep":
        print("Launching mu sweep (3 runs in parallel)...")
        procs = []
        for cfg in MU_SWEEP_CONFIGS:
            p = launch_run(cfg, threads=args.threads)
            if p: procs.append((p, cfg["run_name"]))
            time.sleep(2)  # stagger starts slightly
        if procs:
            print(f"Launched {len(procs)} sweep runs. Waiting for completion...")
            wait_for_completion(procs)
            select_mu_from_sweep()
    
    elif args.phase == "fedavg":
        print("Launching FedAvg runs (6 runs in parallel)...")
        procs = []
        for cfg in FEDAVG_CONFIGS:
            p = launch_run(cfg, threads=args.threads)
            if p: procs.append((p, cfg["run_name"]))
            time.sleep(2)
        if procs:
            print(f"Launched {len(procs)} FedAvg runs.")
            wait_for_completion(procs)
    
    elif args.phase == "fedprox":
        if args.mu is None:
            print("ERROR: --mu required for fedprox phase")
            sys.exit(1)
        print(f"Launching FedProx runs (mu={args.mu}, 6 runs in parallel)...")
        procs = []
        for cfg in fedprox_configs(args.mu):
            p = launch_run(cfg, threads=args.threads)
            if p: procs.append((p, cfg["run_name"]))
            time.sleep(2)
        if procs:
            print(f"Launched {len(procs)} FedProx runs.")
            wait_for_completion(procs)
    
    elif args.phase == "all":
        if args.mu is None:
            print("ERROR: --mu required for all phase")
            sys.exit(1)
        print(f"Launching all 12 final runs (mu={args.mu})...")
        all_final = FEDAVG_CONFIGS + fedprox_configs(args.mu)
        procs = []
        for cfg in all_final:
            p = launch_run(cfg, threads=args.threads)
            if p: procs.append((p, cfg["run_name"]))
            time.sleep(2)
        if procs:
            print(f"Launched {len(procs)} runs.")
            wait_for_completion(procs)

