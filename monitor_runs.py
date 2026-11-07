"""
monitor_runs.py - Show progress of all experiment runs.
Usage: python monitor_runs.py
"""
import json
import os
from pathlib import Path
import time

RESULTS_DIR = Path("results")
ALL_RUN_NAMES = [
    "fedprox_sweep_mu0001", "fedprox_sweep_mu001", "fedprox_sweep_mu01",
    "fedavg_clean_seed42", "fedavg_clean_seed123", "fedavg_clean_seed2026",
    "fedavg_noisy_seed42", "fedavg_noisy_seed123", "fedavg_noisy_seed2026",
    "fedprox_clean_seed42", "fedprox_clean_seed123", "fedprox_clean_seed2026",
    "fedprox_noisy_seed42", "fedprox_noisy_seed123", "fedprox_noisy_seed2026",
]

def check_run(run_name):
    result = {"name": run_name, "status": "not_started", "rounds": 0, "last_val": None, "last_test": None, "last_round_time": None}
    result_dir = RESULTS_DIR / run_name
    
    history_path = result_dir / "history.json"
    stdout_log = result_dir / "stdout.log"
    
    if not result_dir.exists():
        return result
    
    if history_path.exists():
        try:
            h = json.load(open(history_path))
            rounds = h.get("rounds", [])
            result["rounds"] = len(rounds)
            if rounds:
                last = rounds[-1]
                result["last_val"] = last.get("official_val_accuracy", last.get("official_val_acc"))
                result["last_test"] = last.get("official_test_accuracy", last.get("official_test_acc"))
                result["last_round_time"] = last.get("duration_seconds")
            if len(rounds) >= 100:
                result["status"] = "complete"
            else:
                result["status"] = "running" if stdout_log.exists() else "incomplete"
        except Exception as e:
            result["status"] = f"error: {e}"
    elif stdout_log.exists():
        # Read last few lines of stdout log
        try:
            content = stdout_log.read_text(encoding="utf-8", errors="replace")
            lines = [l for l in content.splitlines() if l.strip()]
            # Find latest round info
            round_lines = [l for l in lines if "Round" in l and "/" in l and "Val" in l]
            if round_lines:
                result["status"] = "running"
                result["last_log"] = round_lines[-1].strip()
            elif lines:
                result["status"] = "running"
                result["last_log"] = lines[-1].strip()
        except Exception:
            result["status"] = "running"
    
    return result

def show_status():
    print(f"\n{'='*80}")
    print(f"Experiment Status  [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"{'='*80}")
    
    total = len(ALL_RUN_NAMES)
    complete = 0
    running = 0
    not_started = 0
    
    for run_name in ALL_RUN_NAMES:
        r = check_run(run_name)
        status = r["status"]
        rounds = r["rounds"]
        
        if status == "complete":
            complete += 1
            val_str = f"val={r['last_val']:.4f}" if r['last_val'] is not None else ""
            test_str = f"test={r['last_test']:.4f}" if r['last_test'] is not None else ""
            print(f"  [DONE] {run_name:<35} {rounds:>3}/100 rounds  {val_str}  {test_str}")
        elif status == "running":
            running += 1
            if "last_log" in r:
                print(f"  [RUN ] {run_name:<35} {rounds:>3}/100 rounds  {r.get('last_log', '')[:60]}")
            else:
                val_str = f"val={r['last_val']:.4f}" if r['last_val'] is not None else ""
                print(f"  [RUN ] {run_name:<35} {rounds:>3}/100 rounds  {val_str}")
        elif status == "not_started":
            not_started += 1
            print(f"  [ -- ] {run_name:<35} not started")
        else:
            print(f"  [ !  ] {run_name:<35} {status}")
    
    print(f"\nSummary: {complete}/{total} complete, {running} running, {not_started} not started")
    
    # Check mu selection
    mu_sel_path = RESULTS_DIR / "mu_selection.json"
    if mu_sel_path.exists():
        ms = json.load(open(mu_sel_path))
        print(f"Selected mu: {ms['selected_mu']} (from sweep)")
    else:
        print("mu selection: not yet done")

if __name__ == "__main__":
    show_status()

