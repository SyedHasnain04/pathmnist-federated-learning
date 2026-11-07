"""
collect_results.py - Post-run artifact packaging and collection script.

Collects experiment artifacts from the specified results directory into submission/:
- Diagnostic and convergence plots -> submission/plots/
- Result tables (results.md, all_results.json) -> submission/metrics/
- Mu selection metadata (mu_selection.json) -> submission/metrics/
- Per-run execution histories and confusion matrices -> submission/metrics/
- One-page summary PDF -> submission/summary.pdf
- Skips all large model checkpoints (final_model.pt, checkpoint_model.pt).
- Updates README.md Results section dynamically from empirical metrics.

Usage:
  python collect_results.py --results_dir results
  python collect_results.py --results_dir /path/to/drive/PathMNIST_Results/final
"""
import os
import sys
import shutil
import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Collect experiment artifacts into submission folder.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root results directory containing experiment runs and deliverables.")
    parser.add_argument("--submission_dir", type=str, default="submission",
                        help="Target submission directory (default: submission).")
    parser.add_argument("--update_readme", action="store_true", default=True,
                        help="Update README.md Results section from collected metrics.")
    return parser.parse_args()

def collect_artifacts(results_dir_str, submission_dir_str, update_readme=True):
    results_dir = Path(results_dir_str).resolve()
    submission_dir = Path(submission_dir_str).resolve()
    
    plots_dst = submission_dir / "plots"
    metrics_dst = submission_dir / "metrics"
    plots_dst.mkdir(parents=True, exist_ok=True)
    metrics_dst.mkdir(parents=True, exist_ok=True)

    copied_items = []
    skipped_items = []

    print(f"\n{'='*75}")
    print(f"COLLECTING EXPERIMENT DELIVERABLES")
    print(f"Source results directory: {results_dir}")
    print(f"Destination submission directory: {submission_dir}")
    print(f"{'='*75}\n")

    # 1. Collect Plots
    plots_src = results_dir / "plots"
    if plots_src.exists():
        for pfile in plots_src.glob("*.png"):
            dst_file = plots_dst / pfile.name
            shutil.copy2(pfile, dst_file)
            copied_items.append(f"Plot: {pfile.name} -> {dst_file.relative_to(Path.cwd()) if dst_file.is_relative_to(Path.cwd()) else dst_file}")
    else:
        print(f"Notice: Plots directory not found at {plots_src}")

    # 2. Collect Table files
    tables_src = results_dir / "tables"
    if tables_src.exists():
        for tname in ["results.md", "all_results.json"]:
            tpath = tables_src / tname
            if tpath.exists():
                dst_file = metrics_dst / tname
                shutil.copy2(tpath, dst_file)
                copied_items.append(f"Table: {tname} -> {dst_file.relative_to(Path.cwd()) if dst_file.is_relative_to(Path.cwd()) else dst_file}")

    # 3. Collect mu_selection.json
    mu_file = results_dir / "mu_selection.json"
    if mu_file.exists():
        dst_file = metrics_dst / "mu_selection.json"
        shutil.copy2(mu_file, dst_file)
        copied_items.append(f"Config: mu_selection.json -> {dst_file.relative_to(Path.cwd()) if dst_file.is_relative_to(Path.cwd()) else dst_file}")

    # 4. Collect per-run histories and confusion matrices (strictly skipping large model weights)
    for run_dir in sorted(results_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name not in ["plots", "tables", "wandb"]:
            # Check history.json
            hist_file = run_dir / "history.json"
            if hist_file.exists():
                dst_hist = metrics_dst / f"{run_dir.name}_history.json"
                shutil.copy2(hist_file, dst_hist)
                copied_items.append(f"Run History: {run_dir.name}/history.json -> {dst_hist.name}")
            
            # Check confusion matrix
            cm_file = run_dir / "confusion_matrix.npy"
            if cm_file.exists():
                dst_cm = metrics_dst / f"{run_dir.name}_confusion_matrix.npy"
                shutil.copy2(cm_file, dst_cm)
                copied_items.append(f"Confusion Matrix: {run_dir.name}/confusion_matrix.npy -> {dst_cm.name}")

            # Explicitly identify and skip model checkpoints
            for model_name in ["final_model.pt", "checkpoint_model.pt", "checkpoint_rng.pt"]:
                m_file = run_dir / model_name
                if m_file.exists():
                    skipped_items.append(f"Skipped large file: {run_dir.name}/{model_name} ({m_file.stat().st_size / 1e6:.2f} MB)")

    # 5. Collect Report PDF
    pdf_candidates = [
        results_dir / "PathMNIST_FedAvg_vs_FedProx_Report.pdf",
        results_dir.parent / "PathMNIST_FedAvg_vs_FedProx_Report.pdf",
        Path("PathMNIST_FedAvg_vs_FedProx_Report.pdf").resolve()
    ]
    found_pdf = False
    for pdf_candidate in pdf_candidates:
        if pdf_candidate.exists():
            dst_pdf = submission_dir / "summary.pdf"
            shutil.copy2(pdf_candidate, dst_pdf)
            copied_items.append(f"Report PDF: {pdf_candidate.name} -> {dst_pdf.relative_to(Path.cwd()) if dst_pdf.is_relative_to(Path.cwd()) else dst_pdf}")
            found_pdf = True
            break
    if not found_pdf:
        print("Notice: Report PDF not found yet in candidate paths.")

    # Print summary of actions
    print(f"--- COPIED ARTIFACTS ({len(copied_items)} total) ---")
    for item in copied_items:
        print(f"  [+] {item}")

    print(f"\n--- SKIPPED ARTIFACTS ({len(skipped_items)} total) ---")
    for item in skipped_items:
        print(f"  [-] {item}")

    if update_readme:
        update_readme_results(metrics_dst)

    print(f"\nArtifact collection to {submission_dir} complete!\n")

def update_readme_results(metrics_dir):
    """Dynamically updates the Results section of README.md from empirical metrics."""
    all_results_path = metrics_dir / "all_results.json"
    mu_path = metrics_dir / "mu_selection.json"
    readme_path = Path("README.md")
    
    if not all_results_path.exists() or not readme_path.exists():
        print("Notice: README results dynamic update skipped (results data not yet populated).")
        return

    try:
        with open(all_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mu_data = json.load(open(mu_path, "r", encoding="utf-8")) if mu_path.exists() else {}
        
        selected_mu = mu_data.get("selected_mu", data.get("mu_sweep", {}).get("selected_mu", "N/A"))
        mu_rows = mu_data.get("sweep_results", data.get("mu_sweep", {}).get("rows", []))
        group_summary = data.get("group_summary", [])
        
        # Build dynamic markdown text
        md_lines = ["\n<!-- RESULTS_SECTION_START -->\n"]
        md_lines.append("## 9. Empirical Results & Findings\n")
        md_lines.append(f"### Pre-Experimental μ Sweep (Noisy Condition, Seed 42)\n")
        md_lines.append(f"Optimal parameter selected: **μ = {selected_mu}** (criterion: highest mean official validation accuracy rounds 91–100; ties broken by smaller μ).\n")
        md_lines.append("| Candidate μ | Completed Rounds | Val Acc (91-100 Mean) | Test Acc (91-100 Mean) | Val Acc (Round 100) | Selected |")
        md_lines.append("|---|---|---|---|---|---|")
        for r in mu_rows:
            sel_str = " **YES (Selected)**" if str(r.get("mu")) == str(selected_mu) else ""
            md_lines.append(f"| {r.get('mu')} | {r.get('n_rounds', 100)}/100 | {r.get('val_mean_91_100')} | {r.get('test_mean_91_100')} | {r.get('val_r100')} |{sel_str} |")

        md_lines.append("\n### Final 2×2 Factorial Performance Summary\n")
        md_lines.append("Client 0 aggregation weights by seed: **Seed 42: p0 = 0.1172**, **Seed 123: p0 = 0.3423**, **Seed 2026: p0 = 0.1315**.\n")
        md_lines.append("| Experimental Group | Seed 42 (p0=0.117) | Seed 123 (p0=0.342) | Seed 2026 (p0=0.132) | Test Acc 91-100 (Mean ± Std) | Test Acc (Round 100) | Val Acc (91-100 Mean) | C0 Holdout (Round 100) |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        
        gs_map = {}
        for g in group_summary:
            grp_name = g.get("group", "")
            gs_map[grp_name] = g
            s42 = g.get("seed_42_test_mean_91_100", "N/A")
            s123 = g.get("seed_123_test_mean_91_100", "N/A")
            s2026 = g.get("seed_2026_test_mean_91_100", "N/A")
            t_late = g.get("test_mean_91_100", "N/A")
            t_r100 = g.get("test_r100", "N/A")
            v_late = g.get("val_mean_91_100", "N/A")
            c0_h = g.get("c0_holdout_r100", "N/A")
            md_lines.append(f"| **{grp_name}** | {s42} | {s123} | {s2026} | {t_late} | {t_r100} | {v_late} | {c0_h} |")

        # Dynamic sentence generation from empirical numbers (strictly without pre-baked bias)
        def parse_mean(val_str):
            try:
                return float(val_str.split("±")[0].strip())
            except Exception:
                return None

        fa_noisy_test = parse_mean(gs_map.get("FEDAVG noisy", {}).get("test_mean_91_100", ""))
        fp_noisy_test = parse_mean(gs_map.get("FEDPROX noisy", {}).get("test_mean_91_100", ""))
        fa_clean_test = parse_mean(gs_map.get("FEDAVG clean", {}).get("test_mean_91_100", ""))
        fp_clean_test = parse_mean(gs_map.get("FEDPROX clean", {}).get("test_mean_91_100", ""))

        md_lines.append("\n### Data-Derived Observations (Computed Directly From Metrics)\n")
        if fa_noisy_test is not None and fp_noisy_test is not None:
            diff = fp_noisy_test - fa_noisy_test
            direction = "higher" if diff > 0 else "lower"
            md_lines.append(f"- Under the 40% label noise condition on Client 0, FedProx rounds 91–100 mean test accuracy was **{direction} than FedAvg by {abs(diff):.4f}** (FedProx: {fp_noisy_test:.4f} vs FedAvg: {fa_noisy_test:.4f}).\n")
        
        if fa_clean_test is not None and fa_noisy_test is not None:
            fa_drop = fa_clean_test - fa_noisy_test
            md_lines.append(f"- For FedAvg, the noise degradation (clean vs noisy test accuracy rounds 91–100) was **{fa_drop:+.4f}** (clean: {fa_clean_test:.4f} vs noisy: {fa_noisy_test:.4f}).\n")
            
        if fp_clean_test is not None and fp_noisy_test is not None:
            fp_drop = fp_clean_test - fp_noisy_test
            md_lines.append(f"- For FedProx, the noise degradation (clean vs noisy test accuracy rounds 91–100) was **{fp_drop:+.4f}** (clean: {fp_clean_test:.4f} vs noisy: {fp_noisy_test:.4f}).\n")

        md_lines.append("\n```markdown\n<!-- TODO: my analysis -->\n[Insert author analysis, interpretation of LOO cosine divergences, and clinical conclusions here]\n```\n")
        md_lines.append("<!-- RESULTS_SECTION_END -->\n")

        # Inject into README
        readme_content = readme_path.read_text(encoding="utf-8")
        if "<!-- RESULTS_SECTION_START -->" in readme_content and "<!-- RESULTS_SECTION_END -->" in readme_content:
            start_idx = readme_content.find("<!-- RESULTS_SECTION_START -->")
            end_idx = readme_content.find("<!-- RESULTS_SECTION_END -->") + len("<!-- RESULTS_SECTION_END -->")
            new_readme = readme_content[:start_idx] + "".join(md_lines) + readme_content[end_idx:]
        else:
            new_readme = readme_content + "\n" + "".join(md_lines)
            
        readme_path.write_text(new_readme, encoding="utf-8")
        print("Successfully updated README.md with data-derived Results section!")
    except Exception as e:
        print(f"Warning: Could not update README results: {e}")

if __name__ == "__main__":
    args = parse_args()
    collect_artifacts(args.results_dir, args.submission_dir, update_readme=args.update_readme)

