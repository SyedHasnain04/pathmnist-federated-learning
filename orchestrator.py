"""
orchestrator.py - Master experiment orchestrator.
Manages 2 concurrent worker threads executing the required experimental matrix.
- Worker 1 executes the mu sweep, determines the optimal mu, reuses the winning sweep run as fedprox_noisy_seed42,
  and runs the remaining FedProx runs.
- Worker 2 executes the FedAvg runs and assists with any remaining FedProx runs.
- Automatically handles resumes, logs progress, selects mu, and triggers final tables/plots generation.
"""
import os
import sys
import json
import time
import shutil
import subprocess
import threading
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / "results"
ROUNDS = 100
THREADS_PER_RUN = 4

# Lock for shared queue access and mu selection
lock = threading.Lock()
mu_selected_event = threading.Event()
selected_mu_val = None

def run_is_complete(run_name):
    """Check if a run is already finished with 100 rounds and final_model.pt."""
    h_path = RESULTS_DIR / run_name / "history.json"
    m_path = RESULTS_DIR / run_name / "final_model.pt"
    if not h_path.exists() or not m_path.exists():
        return False
    try:
        h = json.load(open(h_path, encoding="utf-8"))
        return len(h.get("rounds", [])) >= ROUNDS
    except Exception:
        return False

def execute_single_run(config):
    """Execute a single run synchronously via federated_runner.py."""
    run_name = config["run_name"]
    if run_is_complete(run_name):
        print(f"[{time.strftime('%H:%M:%S')}] [SKIP] {run_name} already complete.", flush=True)
        return True

    cmd = [
        sys.executable, "-u", "federated_runner.py",
        "--run_name", run_name,
        "--algorithm", config["algorithm"],
        "--condition", config["condition"],
        "--seed", str(config["seed"]),
        "--mu", str(config.get("mu", 0.0)),
        "--rounds", str(ROUNDS),
        "--threads", str(THREADS_PER_RUN),
    ]

    log_path = RESULTS_DIR / run_name / "stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] [START] {run_name} (algo={config['algorithm']}, cond={config['condition']}, seed={config['seed']}, mu={config.get('mu', 0.0)})", flush=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS_PER_RUN)
    env["MKL_NUM_THREADS"] = str(THREADS_PER_RUN)
    env["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_RUN)

    with open(log_path, "a", encoding="utf-8") as logf:
        ret = subprocess.run(cmd, cwd=str(BASE_DIR), stdout=logf, stderr=subprocess.STDOUT, env=env)

    if ret.returncode == 0 and run_is_complete(run_name):
        print(f"[{time.strftime('%H:%M:%S')}] [SUCCESS] {run_name} completed 100 rounds.", flush=True)
        return True
    else:
        print(f"[{time.strftime('%H:%M:%S')}] [FAILURE] {run_name} exited with code {ret.returncode}", flush=True)
        return False

def evaluate_and_select_mu():
    """Evaluate mu sweep on noisy seed 42 over rounds 91-100."""
    global selected_mu_val
    print("\n" + "="*70, flush=True)
    print("EVALUATING MU SWEEP (rounds 91-100 mean official val accuracy)...", flush=True)
    print("="*70, flush=True)

    sweep_configs = [
        {"run_name": "fedprox_sweep_mu0001", "mu": 0.001},
        {"run_name": "fedprox_sweep_mu001",  "mu": 0.01},
        {"run_name": "fedprox_sweep_mu01",   "mu": 0.1},
    ]

    candidates = {}
    for cfg in sweep_configs:
        name = cfg["run_name"]
        mu = cfg["mu"]
        h_path = RESULTS_DIR / name / "history.json"
        if not h_path.exists():
            raise FileNotFoundError(f"Missing history for sweep run {name}")
        h = json.load(open(h_path, encoding="utf-8"))
        rounds = h.get("rounds", [])
        late_rounds = [r for r in rounds if 91 <= r["round"] <= 100]
        val_accs = [r.get("official_val_accuracy", r.get("official_val_acc", 0.0)) for r in late_rounds]
        mean_val = sum(val_accs) / len(val_accs) if val_accs else 0.0
        candidates[mu] = {
            "run_name": name,
            "mean_val_91_100": mean_val,
            "n_rounds": len(val_accs)
        }
        print(f"  Candidate mu={mu}: Mean Official Val Acc (Rounds 91-100) = {mean_val:.4f}", flush=True)

    # Criterion: max mean val acc; ties broken by smaller mu
    best_mu = max(candidates.keys(), key=lambda m: (candidates[m]["mean_val_91_100"], -m))
    print(f"\n>>> SELECTED MU: {best_mu} (Validation Score: {candidates[best_mu]['mean_val_91_100']:.4f})", flush=True)

    # Save selection metadata
    sel_data = {
        "selected_mu": best_mu,
        "selection_criterion": "highest mean official validation accuracy over rounds 91-100; ties broken by smallest mu",
        "candidates": candidates,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(RESULTS_DIR / "mu_selection.json", "w", encoding="utf-8") as f:
        json.dump(sel_data, f, indent=2)

    # Reuse winning sweep run as fedprox_noisy_seed42
    winning_dir = RESULTS_DIR / candidates[best_mu]["run_name"]
    target_dir = RESULTS_DIR / "fedprox_noisy_seed42"
    if not target_dir.exists():
        print(f"Reusing winning sweep run {winning_dir.name} as fedprox_noisy_seed42...", flush=True)
        shutil.copytree(winning_dir, target_dir)
        # Update run_name in target history.json
        t_hist_path = target_dir / "history.json"
        if t_hist_path.exists():
            th = json.load(open(t_hist_path, encoding="utf-8"))
            th["config"]["run_name"] = "fedprox_noisy_seed42"
            with open(t_hist_path, "w", encoding="utf-8") as f:
                json.dump(th, f, indent=2)
        print("Successfully created fedprox_noisy_seed42 from winning sweep run.", flush=True)

    selected_mu_val = best_mu
    mu_selected_event.set()
    return best_mu

# Shared queue for FedProx runs after mu is selected
fedprox_queue = []

def worker_1_stream():
    """Worker 1: Runs mu sweep, triggers selection, then processes FedProx runs."""
    print("Worker 1 started (Sweep -> FedProx).", flush=True)

    # 1. Run mu sweep
    sweep_runs = [
        {"run_name": "fedprox_sweep_mu0001", "algorithm": "fedprox", "condition": "noisy", "seed": 42, "mu": 0.001},
        {"run_name": "fedprox_sweep_mu001",  "algorithm": "fedprox", "condition": "noisy", "seed": 42, "mu": 0.01},
        {"run_name": "fedprox_sweep_mu01",   "algorithm": "fedprox", "condition": "noisy", "seed": 42, "mu": 0.1},
    ]

    for cfg in sweep_runs:
        execute_single_run(cfg)

    # 2. Select mu and populate FedProx queue
    with lock:
        best_mu = evaluate_and_select_mu()
        # Remaining FedProx runs (fedprox_noisy_seed42 was already populated by reuse)
        fedprox_queue.extend([
            {"run_name": "fedprox_clean_seed42",   "algorithm": "fedprox", "condition": "clean", "seed": 42,   "mu": best_mu},
            {"run_name": "fedprox_clean_seed123",  "algorithm": "fedprox", "condition": "clean", "seed": 123,  "mu": best_mu},
            {"run_name": "fedprox_clean_seed2026", "algorithm": "fedprox", "condition": "clean", "seed": 2026, "mu": best_mu},
            {"run_name": "fedprox_noisy_seed123",  "algorithm": "fedprox", "condition": "noisy", "seed": 123,  "mu": best_mu},
            {"run_name": "fedprox_noisy_seed2026", "algorithm": "fedprox", "condition": "noisy", "seed": 2026, "mu": best_mu},
        ])

    # 3. Pull from fedprox_queue until empty
    while True:
        cfg = None
        with lock:
            if fedprox_queue:
                cfg = fedprox_queue.pop(0)
        if cfg is None:
            break
        execute_single_run(cfg)

    print("Worker 1 finished all assigned runs.", flush=True)

def worker_2_stream():
    """Worker 2: Runs FedAvg runs, then helps with remaining FedProx runs if any."""
    print("Worker 2 started (FedAvg -> assistance).", flush=True)

    fedavg_runs = [
        {"run_name": "fedavg_clean_seed42",   "algorithm": "fedavg", "condition": "clean", "seed": 42,   "mu": 0.0},
        {"run_name": "fedavg_clean_seed123",  "algorithm": "fedavg", "condition": "clean", "seed": 123,  "mu": 0.0},
        {"run_name": "fedavg_clean_seed2026", "algorithm": "fedavg", "condition": "clean", "seed": 2026, "mu": 0.0},
        {"run_name": "fedavg_noisy_seed42",   "algorithm": "fedavg", "condition": "noisy", "seed": 42,   "mu": 0.0},
        {"run_name": "fedavg_noisy_seed123",  "algorithm": "fedavg", "condition": "noisy", "seed": 123,  "mu": 0.0},
        {"run_name": "fedavg_noisy_seed2026", "algorithm": "fedavg", "condition": "noisy", "seed": 2026, "mu": 0.0},
    ]

    for cfg in fedavg_runs:
        execute_single_run(cfg)

    # After FedAvg finishes, wait or assist with FedProx queue
    print("Worker 2 finished FedAvg runs. Checking for FedProx work...", flush=True)
    while True:
        cfg = None
        with lock:
            if fedprox_queue:
                cfg = fedprox_queue.pop(0)
        if cfg is not None:
            execute_single_run(cfg)
        else:
            # If mu is not yet selected, sleep briefly and check again
            if not mu_selected_event.is_set():
                time.sleep(10)
                continue
            else:
                break

    print("Worker 2 finished all assigned runs.", flush=True)

def generate_final_deliverables():
    """Generate tables, plots, and deliverables when all runs are complete."""
    print("\n" + "="*70, flush=True)
    print("ALL RUNS COMPLETE. GENERATING FINAL DELIVERABLES...", flush=True)
    print("="*70, flush=True)

    try:
        subprocess.run([sys.executable, "generate_tables.py"], check=True)
        print("Generated result tables.", flush=True)
    except Exception as e:
        print(f"Error generating tables: {e}", flush=True)

    try:
        subprocess.run([sys.executable, "generate_plots.py"], check=True)
        print("Generated all plots.", flush=True)
    except Exception as e:
        print(f"Error generating plots: {e}", flush=True)

    try:
        subprocess.run([sys.executable, "generate_report_pdf.py"], check=True)
        print("Generated 1-page PDF report.", flush=True)
    except Exception as e:
        print(f"Error generating report PDF: {e}", flush=True)

    print("\nPipeline execution complete!", flush=True)

if __name__ == "__main__":
    print(f"Starting Federated Learning Orchestrator at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: 2 worker streams, {THREADS_PER_RUN} threads per run, {ROUNDS} rounds per run.")

    t1 = threading.Thread(target=worker_1_stream, name="Worker-1")
    t2 = threading.Thread(target=worker_2_stream, name="Worker-2")

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    generate_final_deliverables()
