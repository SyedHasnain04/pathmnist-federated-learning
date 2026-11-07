"""
generate_tables.py - Generate result tables from saved experiment history.json files.

Tables produced:
1. Round 100 metrics for all 12 runs (val acc, test acc, C0 holdout acc)
2. Mean+/-std across 3 seeds for each of 4 groups
3. Last-10-round (91-100) mean for each run
4. mu sweep results table
5. Per-seed detailed results

Usage:
  python generate_tables.py [--results_dir RESULTS_DIR]
"""
import argparse
import json
import numpy as np
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Generate experiment result tables.")
    p.add_argument("--results_dir", default="results",
                   help="Root results directory (default: results)")
    return p.parse_args()

_args = parse_args()
RESULTS_DIR = Path(_args.results_dir)
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
NUM_ROUNDS = 100

GROUPS = [
    ("fedavg", "clean"),
    ("fedavg", "noisy"),
    ("fedprox", "clean"),
    ("fedprox", "noisy"),
]

def load_history(run_name):
    p = RESULTS_DIR / run_name / "history.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None

def get_metric_at_round(h, round_num, key):
    """Get metric value at a specific round."""
    if h is None:
        return None
    for r in h.get("rounds", []):
        if r["round"] == round_num:
            return r.get(key, r.get(key.replace("accuracy", "acc")))
    return None

def get_mean_metric(h, start_round, end_round, key):
    """Get mean metric over round range [start_round, end_round] inclusive."""
    if h is None:
        return None
    vals = []
    for r in h.get("rounds", []):
        if start_round <= r["round"] <= end_round:
            v = r.get(key, r.get(key.replace("accuracy", "acc")))
            if v is not None:
                vals.append(v)
    return np.mean(vals) if vals else None

def get_num_complete_rounds(h):
    if h is None:
        return 0
    return len(h.get("rounds", []))

def format_val(v, fmt=".4f"):
    if v is None:
        return "N/A"
    return format(v, fmt)

def build_run_table():
    """Table 1: Per-run results at round 100 and mean 91-100."""
    rows = []
    
    for algo, cond in GROUPS:
        for seed in SEEDS:
            run_name = f"{algo}_{cond}_seed{seed}"
            h = load_history(run_name)
            n_rounds = get_num_complete_rounds(h)
            
            test_r100 = get_metric_at_round(h, 100, "official_test_accuracy")
            val_r100 = get_metric_at_round(h, 100, "official_val_accuracy")
            test_mean = get_mean_metric(h, 91, 100, "official_test_accuracy")
            val_mean = get_mean_metric(h, 91, 100, "official_val_accuracy")
            c0_holdout = None
            if h:
                for r in h.get("rounds", []):
                    if r["round"] == 100:
                        accs = r.get("client_holdout_accuracies", [])
                        c0_holdout = accs[0] if accs else None
                        break
            
            rows.append({
                "run_name": run_name,
                "algorithm": algo,
                "condition": cond,
                "seed": seed,
                "n_rounds": n_rounds,
                "test_r100": test_r100,
                "val_r100": val_r100,
                "test_mean_91_100": test_mean,
                "val_mean_91_100": val_mean,
                "c0_holdout_r100": c0_holdout,
            })
    
    return rows

def build_group_summary(rows):
    """Table 2: Mean±std across seeds for each group."""
    summary = []
    for algo, cond in GROUPS:
        group_rows = [r for r in rows if r["algorithm"] == algo and r["condition"] == cond]
        
        def agg(key):
            vals = [r[key] for r in group_rows if r[key] is not None]
            if not vals:
                return "N/A", "N/A"
            return np.mean(vals), np.std(vals)
        
        test_mean, test_std = agg("test_r100")
        val_mean, val_std = agg("val_r100")
        test_late_mean, test_late_std = agg("test_mean_91_100")
        val_late_mean, val_late_std = agg("val_mean_91_100")
        c0_mean, c0_std = agg("c0_holdout_r100")
        
        def fmts(m, s):
            if isinstance(m, str):
                return "N/A"
            return f"{m:.4f} ± {s:.4f}"
        
        summary.append({
            "group": f"{algo.upper()} {cond}",
            "n_seeds": len(group_rows),
            "test_r100": fmts(test_mean, test_std),
            "val_r100": fmts(val_mean, val_std),
            "test_mean_91_100": fmts(test_late_mean, test_late_std),
            "val_mean_91_100": fmts(val_late_mean, val_late_std),
            "c0_holdout_r100": fmts(c0_mean, c0_std),
        })
    
    return summary

def build_mu_sweep_table():
    """Table 4: μ sweep results."""
    mu_configs = [
        ("fedprox_sweep_mu0001", 0.001),
        ("fedprox_sweep_mu001", 0.01),
        ("fedprox_sweep_mu01", 0.1),
    ]
    rows = []
    for run_name, mu in mu_configs:
        h = load_history(run_name)
        n_rounds = get_num_complete_rounds(h)
        val_mean_late = get_mean_metric(h, 91, 100, "official_val_accuracy")
        test_mean_late = get_mean_metric(h, 91, 100, "official_test_accuracy")
        val_r100 = get_metric_at_round(h, 100, "official_val_accuracy")
        
        rows.append({
            "mu": mu,
            "run_name": run_name,
            "n_rounds": n_rounds,
            "val_mean_91_100": format_val(val_mean_late),
            "test_mean_91_100": format_val(test_mean_late),
            "val_r100": format_val(val_r100),
        })
    
    # Determine selected mu
    eligible = [(r["mu"], float(r["val_mean_91_100"])) for r in rows if r["val_mean_91_100"] != "N/A"]
    if eligible:
        best_mu = max(eligible, key=lambda x: (x[1], -x[0]))[0]
    else:
        best_mu = "not_selected"
    
    return rows, best_mu

def save_tables(rows, summary, mu_rows, best_mu):
    """Write tables to markdown and JSON files."""
    # --- JSON export ---
    all_data = {
        "per_run_results": rows,
        "group_summary": summary,
        "mu_sweep": {"rows": mu_rows, "selected_mu": best_mu},
    }
    with open(TABLES_DIR / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    print("Saved tables/all_results.json")
    
    # --- Markdown Table 1: Per-run results ---
    md = ["# PathMNIST FedAvg vs FedProx — Experiment Results\n"]
    md.append("## Table 1: Per-Run Results at Round 100\n")
    md.append("| Run Name | Algorithm | Condition | Seed | Rounds | Test@100 | Val@100 | Test Mean 91-100 | Val Mean 91-100 | C0 Holdout@100 |")
    md.append("|----------|-----------|-----------|------|--------|----------|---------|-----------------|-----------------|----------------|")
    for r in rows:
        md.append(f"| {r['run_name']} | {r['algorithm'].upper()} | {r['condition']} | {r['seed']} | "
                  f"{r['n_rounds']}/100 | {format_val(r['test_r100'])} | {format_val(r['val_r100'])} | "
                  f"{format_val(r['test_mean_91_100'])} | {format_val(r['val_mean_91_100'])} | {format_val(r['c0_holdout_r100'])} |")
    
    md.append("\n## Table 2: Group Summary (Mean ± Std across 3 seeds)\n")
    md.append("| Group | N Seeds | Test@100 | Val@100 | Test Mean 91-100 | Val Mean 91-100 | C0 Holdout@100 |")
    md.append("|-------|---------|----------|---------|-----------------|-----------------|----------------|")
    for s in summary:
        md.append(f"| {s['group']} | {s['n_seeds']} | {s['test_r100']} | {s['val_r100']} | "
                  f"{s['test_mean_91_100']} | {s['val_mean_91_100']} | {s['c0_holdout_r100']} |")
    
    md.append(f"\n## Table 3: mu Sweep Results (FedProx, noisy, seed=42)\n")
    md.append(f"**Selected mu = {best_mu}** (criterion: highest mean official val accuracy rounds 91-100; ties broken by smallest mu)\n")
    md.append("| mu | Rounds | Val Mean 91-100 | Test Mean 91-100 | Val@100 | Selected? |")
    md.append("|---|--------|-----------------|-----------------|---------|-----------|")
    for r in mu_rows:
        selected = "YES" if r["mu"] == best_mu else ""
        md.append(f"| {r['mu']} | {r['n_rounds']}/100 | {r['val_mean_91_100']} | {r['test_mean_91_100']} | {r['val_r100']} | {selected} |")
    
    md_text = "\n".join(md)
    with open(TABLES_DIR / "results.md", "w", encoding="utf-8") as f:
        f.write(md_text)
    print("Saved tables/results.md")
    
    return md_text

def main():
    print("Building result tables...")
    rows = build_run_table()
    summary = build_group_summary(rows)
    mu_rows, best_mu = build_mu_sweep_table()
    save_tables(rows, summary, mu_rows, best_mu)
    
    # Print completion status
    print(f"\nCompletion Status:")
    for r in rows:
        status = "COMPLETE" if r["n_rounds"] >= 100 else f"INCOMPLETE ({r['n_rounds']}/100)"
        print(f"  {r['run_name']}: {status}")

if __name__ == "__main__":
    main()

