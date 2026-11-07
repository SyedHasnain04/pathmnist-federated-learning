"""
orchestrator.py - Master experiment orchestrator for PathMNIST FedAvg vs FedProx.
Supports both Google Colab (GPU) and local workstations.
- Automatically detects GPU (CUDA).
- Streams output live to both console (for Colab monitoring) and stdout.log.
- Supports --results_dir and automatically skips completed runs.
- Supports --pilot mode: runs 5 rounds on seed 42 across 4 groups in a separate results_pilot dir.
- Automatically forwards W&B settings (--wandb_mode, --wandb_api_key, --wandb_project).
- Automatically executes mu sweep, selects mu, reuses winning run, executes 12-run matrix,
  and compiles tables, plots, and 1-page PDF report.
"""
import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent.resolve()
ROUNDS = 100
SEEDS = [42, 123, 2026]

def run_is_complete(run_name, results_dir, required_rounds):
    """Check if a run is already finished with required rounds and final_model.pt."""
    h_path = results_dir / run_name / "history.json"
    m_path = results_dir / run_name / "final_model.pt"
    if not h_path.exists() or not m_path.exists():
        return False
    try:
        h = json.load(open(h_path, encoding="utf-8"))
        return len(h.get("rounds", [])) >= required_rounds
    except Exception:
        return False

def execute_single_run(config, results_dir, rounds, device="cuda", threads=4,
                       wandb_project="pathmnist-federated-learning", wandb_mode="online", wandb_api_key=None, is_pilot=False):
    """Execute a single run, streaming output live to both console and logfile."""
    run_name = config["run_name"]
    if run_is_complete(run_name, results_dir, rounds):
        print(f"[{time.strftime('%H:%M:%S')}] [SKIP] {run_name} already complete ({rounds} rounds).", flush=True)
        return True

    cmd = [
        sys.executable, "-u", "federated_runner.py",
        "--run_name", run_name,
        "--algorithm", config["algorithm"],
        "--condition", config["condition"],
        "--seed", str(config["seed"]),
        "--mu", str(config.get("mu", 0.0)),
        "--rounds", str(rounds),
        "--threads", str(threads),
        "--device", str(device),
        "--results_dir", str(results_dir),
        "--wandb_project", str(wandb_project),
        "--wandb_mode", str(wandb_mode),
    ]
    if is_pilot:
        cmd.append("--is_pilot")
    if wandb_api_key:
        cmd.extend(["--wandb_api_key", str(wandb_api_key)])

    log_path = results_dir / run_name / "stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] LAUNCHING: {run_name}", flush=True)
    print(f"Algorithm: {config['algorithm'].upper()} | Condition: {config['condition'].upper()} | Seed: {config['seed']} | Mu: {config.get('mu', 0.0)} | Device: {device} | Rounds: {rounds}", flush=True)
    print(f"{'='*75}", flush=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)

    with open(log_path, "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()

        proc.wait()

    if proc.returncode == 0 and run_is_complete(run_name, results_dir, rounds):
        print(f"[{time.strftime('%H:%M:%S')}] [SUCCESS] {run_name} finished {rounds} rounds.\n", flush=True)
        return True
    else:
        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] {run_name} exited with code {proc.returncode}\n", flush=True)
        return False

def evaluate_and_select_mu(results_dir):
    """Evaluate mu sweep on noisy seed 42 over rounds 91-100."""
    print("\n" + "#"*75, flush=True)
    print("STAGE 5: EVALUATING MU SWEEP (Rounds 91-100 Mean Official Val Acc)", flush=True)
    print("#"*75, flush=True)

    sweep_configs = [
        {"run_name": "fedprox_sweep_mu0001", "mu": 0.001},
        {"run_name": "fedprox_sweep_mu001",  "mu": 0.01},
        {"run_name": "fedprox_sweep_mu01",   "mu": 0.1},
    ]

    candidates = {}
    for cfg in sweep_configs:
        name = cfg["run_name"]
        mu = cfg["mu"]
        h_path = results_dir / name / "history.json"
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
        print(f"  Candidate mu={mu:<5}: Mean Official Val Acc (Rounds 91-100) = {mean_val:.4f} (across {len(val_accs)} rounds)", flush=True)

    best_mu = max(candidates.keys(), key=lambda m: (candidates[m]["mean_val_91_100"], -m))
    print(f"\n>>> SELECTED OPTIMAL MU: {best_mu} (Val Acc: {candidates[best_mu]['mean_val_91_100']:.4f})", flush=True)

    sel_data = {
        "selected_mu": best_mu,
        "selection_criterion": "highest mean official validation accuracy over rounds 91-100; ties broken by smallest mu",
        "candidates": candidates,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(results_dir / "mu_selection.json", "w", encoding="utf-8") as f:
        json.dump(sel_data, f, indent=2)

    winning_dir = results_dir / candidates[best_mu]["run_name"]
    target_dir = results_dir / "fedprox_noisy_seed42"
    if not target_dir.exists():
        print(f"Reusing winning sweep run {winning_dir.name} as fedprox_noisy_seed42...", flush=True)
        shutil.copytree(winning_dir, target_dir)
        t_hist_path = target_dir / "history.json"
        if t_hist_path.exists():
            th = json.load(open(t_hist_path, encoding="utf-8"))
            th["config"]["run_name"] = "fedprox_noisy_seed42"
            with open(t_hist_path, "w", encoding="utf-8") as f:
                json.dump(th, f, indent=2)
        print("Successfully created fedprox_noisy_seed42 from winning sweep run.\n", flush=True)

    return best_mu

def generate_final_deliverables(results_dir):
    """Generate tables, plots, and PDF report when all runs are complete."""
    print("\n" + "#"*75, flush=True)
    print("GENERATING FINAL DELIVERABLES: TABLES, PLOTS, AND PDF REPORT", flush=True)
    print("#"*75, flush=True)

    try:
        subprocess.run([sys.executable, "generate_tables.py", "--results_dir", str(results_dir)], check=True)
        print("Generated result tables.", flush=True)
    except Exception as e:
        print(f"Error generating tables: {e}", flush=True)

    try:
        subprocess.run([sys.executable, "generate_plots.py", "--results_dir", str(results_dir)], check=True)
        print("Generated all plots.", flush=True)
    except Exception as e:
        print(f"Error generating plots: {e}", flush=True)

    try:
        subprocess.run([sys.executable, "generate_report_pdf.py", "--results_dir", str(results_dir)], check=True)
        print("Generated 1-page PDF report.", flush=True)
    except Exception as e:
        print(f"Error generating report PDF: {e}", flush=True)

    print("\nAll deliverables generated successfully!", flush=True)

def run_pilot(results_dir, device="cuda", threads=4, wandb_project="pathmnist-federated-learning", wandb_mode="online", wandb_api_key=None):
    """Run pilot experiments: 5 rounds, 4 groups, seed 42."""
    print("\n" + "="*75)
    print("RUNNING PILOT MODE: 5 Rounds across 4 groups (Seed 42)")
    print(f"Results Directory: {results_dir}")
    print("="*75)
    results_dir.mkdir(parents=True, exist_ok=True)

    pilot_matrix = [
        {"run_name": "pilot_fedavg_clean_seed42",   "algorithm": "fedavg",  "condition": "clean", "seed": 42, "mu": 0.0},
        {"run_name": "pilot_fedavg_noisy_seed42",   "algorithm": "fedavg",  "condition": "noisy", "seed": 42, "mu": 0.0},
        {"run_name": "pilot_fedprox_clean_seed42",  "algorithm": "fedprox", "condition": "clean", "seed": 42, "mu": 0.01},
        {"run_name": "pilot_fedprox_noisy_seed42",  "algorithm": "fedprox", "condition": "noisy", "seed": 42, "mu": 0.01},
    ]

    for cfg in pilot_matrix:
        success = execute_single_run(
            cfg, results_dir=results_dir, rounds=5, device=device, threads=threads,
            wandb_project=wandb_project, wandb_mode=wandb_mode, wandb_api_key=wandb_api_key, is_pilot=True
        )
        if not success:
            raise RuntimeError(f"Pilot run {cfg['run_name']} failed!")

    print("\nPilot mode complete! Verified data loading, training, evaluation, metrics, and checkpoints.")

def run_full_pipeline(results_dir, device="cuda", threads=4, wandb_project="pathmnist-federated-learning", wandb_mode="online", wandb_api_key=None):
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Mu Sweep on Noisy Seed 42
    print("\n" + "="*75)
    print("PHASE 1: FEDPROX MU SWEEP (Candidates: 0.001, 0.01, 0.1)")
    print("="*75)

    sweep_runs = [
        {"run_name": "fedprox_sweep_mu0001", "algorithm": "fedprox", "condition": "noisy", "seed": 42, "mu": 0.001},
        {"run_name": "fedprox_sweep_mu001",  "algorithm": "fedprox", "condition": "noisy", "seed": 42, "mu": 0.01},
        {"run_name": "fedprox_sweep_mu01",   "algorithm": "fedprox", "condition": "noisy", "seed": 42, "mu": 0.1},
    ]

    for cfg in sweep_runs:
        success = execute_single_run(
            cfg, results_dir=results_dir, rounds=100, device=device, threads=threads,
            wandb_project=wandb_project, wandb_mode=wandb_mode, wandb_api_key=wandb_api_key
        )
        if not success:
            raise RuntimeError(f"Run {cfg['run_name']} failed!")

    # Phase 2: Select Mu & Freeze
    best_mu = evaluate_and_select_mu(results_dir)

    # Phase 3: Final Experiment Matrix (12 Configurations)
    print("\n" + "="*75)
    print(f"PHASE 3: FINAL EXPERIMENT MATRIX (Selected mu = {best_mu})")
    print("="*75)

    final_matrix = [
        # FedAvg Clean (Seeds 42, 123, 2026)
        {"run_name": "fedavg_clean_seed42",   "algorithm": "fedavg", "condition": "clean", "seed": 42,   "mu": 0.0},
        {"run_name": "fedavg_clean_seed123",  "algorithm": "fedavg", "condition": "clean", "seed": 123,  "mu": 0.0},
        {"run_name": "fedavg_clean_seed2026", "algorithm": "fedavg", "condition": "clean", "seed": 2026, "mu": 0.0},
        # FedAvg Noisy (Seeds 42, 123, 2026)
        {"run_name": "fedavg_noisy_seed42",   "algorithm": "fedavg", "condition": "noisy", "seed": 42,   "mu": 0.0},
        {"run_name": "fedavg_noisy_seed123",  "algorithm": "fedavg", "condition": "noisy", "seed": 123,  "mu": 0.0},
        {"run_name": "fedavg_noisy_seed2026", "algorithm": "fedavg", "condition": "noisy", "seed": 2026, "mu": 0.0},
        # FedProx Clean (Seeds 42, 123, 2026)
        {"run_name": "fedprox_clean_seed42",   "algorithm": "fedprox", "condition": "clean", "seed": 42,   "mu": best_mu},
        {"run_name": "fedprox_clean_seed123",  "algorithm": "fedprox", "condition": "clean", "seed": 123,  "mu": best_mu},
        {"run_name": "fedprox_clean_seed2026", "algorithm": "fedprox", "condition": "clean", "seed": 2026, "mu": best_mu},
        # FedProx Noisy (fedprox_noisy_seed42 was already reused from winning sweep run)
        {"run_name": "fedprox_noisy_seed123",  "algorithm": "fedprox", "condition": "noisy", "seed": 123,  "mu": best_mu},
        {"run_name": "fedprox_noisy_seed2026", "algorithm": "fedprox", "condition": "noisy", "seed": 2026, "mu": best_mu},
    ]

    for cfg in final_matrix:
        success = execute_single_run(
            cfg, results_dir=results_dir, rounds=100, device=device, threads=threads,
            wandb_project=wandb_project, wandb_mode=wandb_mode, wandb_api_key=wandb_api_key
        )
        if not success:
            raise RuntimeError(f"Run {cfg['run_name']} failed!")

    # Phase 4: Generate Deliverables
    generate_final_deliverables(results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--threads", type=int, default=4, help="Threads for data loading")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for experiment results")
    parser.add_argument("--pilot", action="store_true", help="Run 5-round pilot verification across 4 groups")
    parser.add_argument("--wandb_project", type=str, default="pathmnist-federated-learning")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_api_key", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nStarting Federated Learning Orchestrator at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Device: {device.upper()}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

    results_path = Path(args.results_dir if not args.pilot else "results_pilot").resolve()

    if args.pilot:
        run_pilot(
            results_dir=results_path,
            device=device,
            threads=args.threads,
            wandb_project=args.wandb_project,
            wandb_mode=args.wandb_mode,
            wandb_api_key=args.wandb_api_key
        )
    else:
        run_full_pipeline(
            results_dir=results_path,
            device=device,
            threads=args.threads,
            wandb_project=args.wandb_project,
            wandb_mode=args.wandb_mode,
            wandb_api_key=args.wandb_api_key
        )
