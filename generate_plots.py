"""
generate_plots.py - Generate all required experiment plots from saved history.json files.

Required plots:
1. Official test accuracy vs rounds (4 curves: FedAvg/FedProx × clean/noisy, mean±std across seeds)
2. Official test loss vs rounds
3. Per-client holdout accuracy vs rounds (5 clients, example seed)
4. Official validation accuracy vs rounds
5. Per-client update norms vs rounds
6. Per-client standard cosine similarity vs rounds
7. Per-client leave-one-out cosine similarity vs rounds
8. Client class-distribution heatmap (5×9) from partition metadata
9. Client aggregation weights bar chart
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
NUM_ROUNDS = 100
COLORS = {
    "fedavg_clean": "#1f77b4",
    "fedavg_noisy": "#aec7e8",
    "fedprox_clean": "#d62728",
    "fedprox_noisy": "#ff9896",
}
LABELS = {
    "fedavg_clean": "FedAvg Clean",
    "fedavg_noisy": "FedAvg Noisy",
    "fedprox_clean": "FedProx Clean",
    "fedprox_noisy": "FedProx Noisy",
}
CLIENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def load_run(run_name, key="official_test_accuracy"):
    """Load a specific metric for all rounds of a run."""
    history_path = RESULTS_DIR / run_name / "history.json"
    if not history_path.exists():
        return None
    h = json.load(open(history_path))
    rounds = h.get("rounds", [])
    if not rounds:
        return None
    vals = []
    for r in rounds:
        v = r.get(key, r.get(key.replace("accuracy", "acc"), None))
        vals.append(v if v is not None else float("nan"))
    return np.array(vals)

def load_all_seeds(algorithm, condition, key):
    """Load metric for all 3 seeds, return array (n_seeds, n_rounds)."""
    all_data = []
    for seed in SEEDS:
        run_name = f"{algorithm}_{condition}_seed{seed}"
        vals = load_run(run_name, key)
        if vals is not None and len(vals) == NUM_ROUNDS:
            all_data.append(vals)
        else:
            n = len(vals) if vals is not None else 0
            print(f"  WARNING: {run_name} has {n}/{NUM_ROUNDS} rounds for key={key}")
    if not all_data:
        return None
    return np.array(all_data)

def plot_mean_std(ax, data, label, color, rounds=None):
    """Plot mean ± std across seeds."""
    if data is None or len(data) == 0:
        return
    if rounds is None:
        rounds = np.arange(1, data.shape[1] + 1)
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    ax.plot(rounds, mean, color=color, label=label, linewidth=1.8)
    ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15)

def plot1_test_accuracy():
    """Plot 1: Official test accuracy vs rounds (4 group mean±std curves)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = np.arange(1, NUM_ROUNDS + 1)
    
    for algo in ["fedavg", "fedprox"]:
        for cond in ["clean", "noisy"]:
            group_key = f"{algo}_{cond}"
            data = load_all_seeds(algo, cond, "official_test_accuracy")
            plot_mean_std(ax, data, LABELS[group_key], COLORS[group_key], rounds)
    
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Official Test Accuracy")
    ax.set_title("PathMNIST Cross-Site Test Accuracy vs Rounds\n(Mean ± Std across 3 seeds)")
    ax.legend(loc="lower right")
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot1_test_accuracy_vs_rounds.png", dpi=150)
    plt.close(fig)
    print("Saved plot1_test_accuracy_vs_rounds.png")

def plot2_test_loss():
    """Plot 2: Official test loss vs rounds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = np.arange(1, NUM_ROUNDS + 1)
    
    for algo in ["fedavg", "fedprox"]:
        for cond in ["clean", "noisy"]:
            group_key = f"{algo}_{cond}"
            data = load_all_seeds(algo, cond, "official_test_loss")
            plot_mean_std(ax, data, LABELS[group_key], COLORS[group_key], rounds)
    
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Official Test Loss (Cross-Entropy)")
    ax.set_title("PathMNIST Cross-Site Test Loss vs Rounds\n(Mean ± Std across 3 seeds)")
    ax.legend(loc="upper right")
    ax.set_xlim(1, NUM_ROUNDS)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot2_test_loss_vs_rounds.png", dpi=150)
    plt.close(fig)
    print("Saved plot2_test_loss_vs_rounds.png")

def plot3_client_holdout_accuracy(algo="fedavg", cond="clean", seed=42):
    """Plot 3: Per-client holdout accuracy vs rounds (example run)."""
    run_name = f"{algo}_{cond}_seed{seed}"
    history_path = RESULTS_DIR / run_name / "history.json"
    if not history_path.exists():
        print(f"WARNING: {run_name} not available for plot3")
        return
    
    h = json.load(open(history_path))
    rounds_data = h.get("rounds", [])
    if not rounds_data:
        return
    
    num_clients = 5
    client_accs = [[] for _ in range(num_clients)]
    rounds = []
    for r in rounds_data:
        rounds.append(r["round"])
        for cid in range(num_clients):
            acc = r.get("client_holdout_accuracies", [None]*num_clients)[cid]
            client_accs[cid].append(acc if acc is not None else float("nan"))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for cid in range(num_clients):
        label = f"Client {cid}" + (" (Noisy)" if cid == 0 else "")
        ax.plot(rounds, client_accs[cid], color=CLIENT_COLORS[cid], label=label, linewidth=1.5,
                linestyle="--" if cid == 0 else "-")
    
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Local Holdout Accuracy")
    ax.set_title(f"Per-Client Local Holdout Accuracy vs Rounds\n({run_name})")
    ax.legend(loc="lower right")
    ax.set_xlim(1, max(rounds) if rounds else NUM_ROUNDS)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"plot3_client_holdout_{run_name}.png", dpi=150)
    plt.close(fig)
    print(f"Saved plot3_client_holdout_{run_name}.png")

def plot4_val_accuracy():
    """Plot 4: Official validation accuracy vs rounds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = np.arange(1, NUM_ROUNDS + 1)
    
    for algo in ["fedavg", "fedprox"]:
        for cond in ["clean", "noisy"]:
            group_key = f"{algo}_{cond}"
            data = load_all_seeds(algo, cond, "official_val_accuracy")
            plot_mean_std(ax, data, LABELS[group_key], COLORS[group_key], rounds)
    
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Official Validation Accuracy")
    ax.set_title("PathMNIST Official Validation Accuracy vs Rounds\n(Mean ± Std across 3 seeds)")
    ax.legend(loc="lower right")
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot4_val_accuracy_vs_rounds.png", dpi=150)
    plt.close(fig)
    print("Saved plot4_val_accuracy_vs_rounds.png")

def plot5_update_norms(algo="fedavg", cond="clean", seed=42):
    """Plot 5: Per-client update norms vs rounds."""
    run_name = f"{algo}_{cond}_seed{seed}"
    history_path = RESULTS_DIR / run_name / "history.json"
    if not history_path.exists():
        print(f"WARNING: {run_name} not available for plot5")
        return
    
    h = json.load(open(history_path))
    rounds_data = h.get("rounds", [])
    if not rounds_data:
        return
    
    num_clients = 5
    client_norms = [[] for _ in range(num_clients)]
    rounds = []
    for r in rounds_data:
        rounds.append(r["round"])
        norms = r.get("update_norms", [float("nan")]*num_clients)
        for cid in range(num_clients):
            client_norms[cid].append(norms[cid] if cid < len(norms) else float("nan"))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for cid in range(num_clients):
        label = f"Client {cid}" + (" (Noisy)" if cid == 0 else "")
        ax.plot(rounds, client_norms[cid], color=CLIENT_COLORS[cid], label=label, linewidth=1.5,
                linestyle="--" if cid == 0 else "-")
    
    ax.set_xlabel("Communication Round")
    ax.set_ylabel(r"Update L2 Norm $\|\Delta_k\|_2$")
    ax.set_title(f"Per-Client Update L2 Norms vs Rounds\n({run_name})")
    ax.legend(loc="upper right")
    ax.set_xlim(1, max(rounds) if rounds else NUM_ROUNDS)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"plot5_update_norms_{run_name}.png", dpi=150)
    plt.close(fig)
    print(f"Saved plot5_update_norms_{run_name}.png")

def plot6_standard_cosine(algo="fedavg", cond="noisy", seed=42):
    """Plot 6: Per-client standard cosine similarity vs rounds."""
    run_name = f"{algo}_{cond}_seed{seed}"
    history_path = RESULTS_DIR / run_name / "history.json"
    if not history_path.exists():
        print(f"WARNING: {run_name} not available for plot6")
        return
    
    h = json.load(open(history_path))
    rounds_data = h.get("rounds", [])
    if not rounds_data:
        return
    
    num_clients = 5
    client_cos = [[] for _ in range(num_clients)]
    rounds = []
    for r in rounds_data:
        rounds.append(r["round"])
        cosines = r.get("standard_cosines", [float("nan")]*num_clients)
        for cid in range(num_clients):
            client_cos[cid].append(cosines[cid] if cid < len(cosines) else float("nan"))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for cid in range(num_clients):
        label = f"Client {cid}" + (" (Noisy)" if cid == 0 else "")
        vals = [v if not (isinstance(v, float) and np.isnan(v)) else None for v in client_cos[cid]]
        ax.plot(rounds, client_cos[cid], color=CLIENT_COLORS[cid], label=label, linewidth=1.5,
                linestyle="--" if cid == 0 else "-")
    
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel(r"$\cos(\Delta_k, \Delta_{global})$")
    ax.set_title(f"Per-Client Standard Cosine Similarity vs Rounds\n({run_name})")
    ax.legend(loc="lower right")
    ax.set_xlim(1, max(rounds) if rounds else NUM_ROUNDS)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"plot6_standard_cosine_{run_name}.png", dpi=150)
    plt.close(fig)
    print(f"Saved plot6_standard_cosine_{run_name}.png")

def plot7_loo_cosine(algo="fedavg", cond="noisy", seed=42):
    """Plot 7: Per-client leave-one-out cosine similarity vs rounds."""
    run_name = f"{algo}_{cond}_seed{seed}"
    history_path = RESULTS_DIR / run_name / "history.json"
    if not history_path.exists():
        print(f"WARNING: {run_name} not available for plot7")
        return
    
    h = json.load(open(history_path))
    rounds_data = h.get("rounds", [])
    if not rounds_data:
        return
    
    num_clients = 5
    client_loo = [[] for _ in range(num_clients)]
    rounds = []
    for r in rounds_data:
        rounds.append(r["round"])
        loo_cosines = r.get("leave_one_out_cosines", [float("nan")]*num_clients)
        for cid in range(num_clients):
            client_loo[cid].append(loo_cosines[cid] if cid < len(loo_cosines) else float("nan"))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for cid in range(num_clients):
        label = f"Client {cid}" + (" (Noisy)" if cid == 0 else "")
        ax.plot(rounds, client_loo[cid], color=CLIENT_COLORS[cid], label=label, linewidth=1.5,
                linestyle="--" if cid == 0 else "-")
    
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel(r"$\cos(\Delta_k, \Delta_{-k})$")
    ax.set_title(f"Per-Client Leave-One-Out Cosine Similarity vs Rounds\n({run_name})")
    ax.legend(loc="lower right")
    ax.set_xlim(1, max(rounds) if rounds else NUM_ROUNDS)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"plot7_loo_cosine_{run_name}.png", dpi=150)
    plt.close(fig)
    print(f"Saved plot7_loo_cosine_{run_name}.png")

def plot8_class_distribution_heatmap(seed=42):
    """Plot 8: Client class-distribution heatmap (5×9) from partition metadata."""
    # Load class distribution from partition data
    partition_base = Path(f"data/partitions/seed_{seed}/clean/train")
    class_counts = np.zeros((5, 9), dtype=np.int64)
    
    for cid in range(5):
        npz_path = partition_base / f"{cid}.npz"
        if npz_path.exists():
            d = np.load(npz_path, allow_pickle=True)["data"].item()
            y = d["y"]
            for cls in range(9):
                class_counts[cid, cls] = np.sum(y == cls)
    
    # Normalize to proportions
    row_sums = class_counts.sum(axis=1, keepdims=True)
    proportions = class_counts / row_sums.clip(min=1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(proportions, cmap="Blues", aspect="auto", vmin=0, vmax=0.5)
    
    # Annotate with counts
    for i in range(5):
        for j in range(9):
            ax.text(j, i, f"{class_counts[i, j]}", ha="center", va="center", fontsize=9,
                    color="white" if proportions[i, j] > 0.3 else "black")
    
    class_names = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    ax.set_xticks(range(9))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"Client {i}" + (" (Noisy)" if i == 0 else "") for i in range(5)])
    ax.set_xlabel("PathMNIST Class")
    ax.set_title(f"Client Class Distribution (seed={seed}, α=0.3 Dirichlet)\nNumbers show sample counts, shading shows proportion")
    plt.colorbar(im, ax=ax, label="Class Proportion")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"plot8_class_distribution_seed{seed}.png", dpi=150)
    plt.close(fig)
    print(f"Saved plot8_class_distribution_seed{seed}.png")

def plot9_aggregation_weights():
    """Plot 9: Client aggregation weights bar chart for all 3 seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for idx, seed in enumerate(SEEDS):
        ax = axes[idx]
        # Load weights from any run with this seed
        run_name = f"fedavg_clean_seed{seed}"
        history_path = RESULTS_DIR / run_name / "history.json"
        
        weights = None
        if history_path.exists():
            h = json.load(open(history_path))
            weights = h.get("config", {}).get("aggregation_weights")
        
        if weights is None:
            # Fallback: compute from partition data
            counts = []
            for cid in range(5):
                npz_path = Path(f"data/partitions/seed_{seed}/clean/train/{cid}.npz")
                if npz_path.exists():
                    d = np.load(npz_path, allow_pickle=True)["data"].item()
                    counts.append(len(d["y"]))
                else:
                    counts.append(0)
            total = sum(counts)
            weights = [c/total for c in counts]
        
        client_labels = [f"C{i}" + ("\n(noisy)" if i == 0 else "") for i in range(5)]
        bar_colors = [CLIENT_COLORS[i] for i in range(5)]
        bars = ax.bar(client_labels, weights, color=bar_colors, edgecolor="black", linewidth=0.5)
        
        for bar, w in zip(bars, weights):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{w:.3f}", ha="center", va="bottom", fontsize=9)
        
        ax.set_title(f"Seed {seed}")
        ax.set_ylabel("Aggregation Weight" if idx == 0 else "")
        ax.set_ylim(0, max(weights) * 1.25)
        ax.grid(axis="y", alpha=0.3)
    
    fig.suptitle("Client Aggregation Weights (sample-count weighted)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot9_aggregation_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot9_aggregation_weights.png")

def plot_mu_sweep():
    """Extra: mu sweep results (val accuracy per candidate)."""
    mu_candidates = [0.001, 0.01, 0.1]
    run_names = ["fedprox_sweep_mu0001", "fedprox_sweep_mu001", "fedprox_sweep_mu01"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = np.arange(1, NUM_ROUNDS + 1)
    colors_mu = ["#2ca02c", "#ff7f0e", "#d62728"]
    
    for mu, run_name, color in zip(mu_candidates, run_names, colors_mu):
        vals = load_run(run_name, "official_val_accuracy")
        if vals is not None and len(vals) >= 10:
            # Pad if less than 100 rounds
            x = np.arange(1, len(vals) + 1)
            ax.plot(x, vals, label=f"μ={mu}", color=color, linewidth=1.8)
            # Highlight rounds 91-100
            if len(vals) >= 100:
                mean_late = np.nanmean(vals[90:100])
                ax.axhline(mean_late, color=color, linestyle=":", alpha=0.5)
                ax.text(NUM_ROUNDS + 1, mean_late, f"  {mean_late:.4f}", color=color, fontsize=9, va="center")
    
    # Shade rounds 91-100 region
    ax.axvspan(91, 100, alpha=0.08, color="gray", label="Selection window (rounds 91-100)")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Official Validation Accuracy")
    ax.set_title("FedProx μ Sweep: Official Validation Accuracy\n(noisy condition, seed=42)")
    ax.legend(loc="lower right")
    ax.set_xlim(1, NUM_ROUNDS + 5)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot_mu_sweep.png", dpi=150)
    plt.close(fig)
    print("Saved plot_mu_sweep.png")

def generate_all_plots():
    print(f"\nGenerating plots to: {PLOTS_DIR}")
    
    # Plots 1-2, 4: aggregate across all 12 runs
    print("\n--- Plot 1: Test accuracy vs rounds ---")
    plot1_test_accuracy()
    
    print("\n--- Plot 2: Test loss vs rounds ---")
    plot2_test_loss()
    
    print("\n--- Plot 4: Val accuracy vs rounds ---")
    plot4_val_accuracy()
    
    # Plots 3, 5-7: per-client metrics for representative runs
    print("\n--- Plot 3: Client holdout accuracy (FedAvg noisy seed=42) ---")
    plot3_client_holdout_accuracy("fedavg", "noisy", 42)
    
    # Also generate for FedProx noisy
    print("\n--- Plot 3b: Client holdout accuracy (FedProx noisy seed=42) ---")
    plot3_client_holdout_accuracy("fedprox", "noisy", 42)
    
    print("\n--- Plot 5: Update norms (FedAvg noisy seed=42) ---")
    plot5_update_norms("fedavg", "noisy", 42)
    
    print("\n--- Plot 6: Standard cosine (FedAvg noisy seed=42) ---")
    plot6_standard_cosine("fedavg", "noisy", 42)
    
    print("\n--- Plot 7: LOO cosine (FedAvg noisy seed=42) ---")
    plot7_loo_cosine("fedavg", "noisy", 42)
    
    # Also generate update metrics for FedProx noisy
    print("\n--- Plot 5b: Update norms (FedProx noisy seed=42) ---")
    plot5_update_norms("fedprox", "noisy", 42)
    
    print("\n--- Plot 6b: Standard cosine (FedProx noisy seed=42) ---")
    plot6_standard_cosine("fedprox", "noisy", 42)
    
    print("\n--- Plot 7b: LOO cosine (FedProx noisy seed=42) ---")
    plot7_loo_cosine("fedprox", "noisy", 42)
    
    # Plots 8-9: data distribution
    print("\n--- Plot 8: Class distribution heatmap ---")
    for seed in SEEDS:
        plot8_class_distribution_heatmap(seed)
    
    print("\n--- Plot 9: Aggregation weights ---")
    plot9_aggregation_weights()
    
    # Mu sweep
    print("\n--- Mu sweep plot ---")
    plot_mu_sweep()
    
    print(f"\nAll plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    generate_all_plots()

