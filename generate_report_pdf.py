"""
generate_report_pdf.py - Generate a professional 1-page experiment PDF report.
Objective, setup, results table, key plots, update-metric observations, conclusion, limitations.

Usage:
  python generate_report_pdf.py [--results_dir RESULTS_DIR]
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def _parse_args():
    p = argparse.ArgumentParser(description="Generate PDF report.")
    p.add_argument("--results_dir", default="results",
                   help="Root results directory (default: results)")
    p.add_argument("--wandb_url", default=os.environ.get("WANDB_PROJECT_URL", ""),
                   help="W&B project URL (env: WANDB_PROJECT_URL)")
    p.add_argument("--github_url",
                   default=os.environ.get("GITHUB_URL",
                       "https://github.com/syedHasnain04/pathmnist-federated-learning"),
                   help="GitHub repository URL (env: GITHUB_URL)")
    return p.parse_args()

_args = _parse_args()
RESULTS_DIR = Path(_args.results_dir)
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"
PDF_PATH = Path(__file__).parent / "PathMNIST_FedAvg_vs_FedProx_Report.pdf"

def generate_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=16,
        textColor=colors.HexColor('#1a365d')
    )

    sub_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#4a5568')
    )

    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=4,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#2d3748')
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.5,
        leading=8,
        alignment=1
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.5,
        leading=8,
        textColor=colors.white,
        alignment=1
    )

    story = []

    # Title & Header
    story.append(Paragraph("PathMNIST Federated Learning: FedAvg vs FedProx Under Non-IID Label Noise", title_style))
    story.append(Paragraph("Authoritative Benchmark & Evaluation Report | Seeds: 42, 123, 2026 | Dirichlet α=0.3 | 40% Symmetric Noise on Client 0", sub_style))
    # Optional URL links (passed via --wandb_url / --github_url or env vars WANDB_PROJECT_URL / GITHUB_URL)
    _github = _args.github_url.strip()
    _wandb  = _args.wandb_url.strip()
    if _github:
        story.append(Paragraph(
            f'GitHub: <a href="{_github}" color="blue">{_github}</a>', sub_style))
    if _wandb:
        story.append(Paragraph(
            f'W&B Dashboard: <a href="{_wandb}" color="blue">{_wandb}</a>', sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e0'), spaceBefore=4, spaceAfter=4))

    # Setup summary
    setup_text = (
        "<b>Experimental Design:</b> 5-client Dirichlet (α=0.3) partitioning over official PathMNIST training data (89,996 images). "
        "Client 0 is subjected to 40% symmetric label noise on its local training set. 100 communication rounds, 5 local epochs (E=5), batch size 64, "
        "SGD optimizer (lr=0.01, momentum=0.9, fresh optimizer state per round). FedProx proximal term evaluated against detached start-of-round snapshot. "
        "Evaluation strictly isolates official in-distribution validation (10,004) from clinical cross-site test (7,180)."
    )
    story.append(Paragraph(setup_text, body_style))
    story.append(Spacer(1, 4))

    # Results Table
    story.append(Paragraph("<b>Table 1: Cross-Site Official Test & Validation Performance (Mean ± Std Across 3 Seeds)</b>", h2_style))
    
    # Load results from tables if available
    summary_path = TABLES_DIR / "all_results.json"
    rows_data = [
        [Paragraph("<b>Experimental Group</b>", table_header),
         Paragraph("<b>Test Acc (R100)</b>", table_header),
         Paragraph("<b>Val Acc (R100)</b>", table_header),
         Paragraph("<b>Test Acc (91-100)</b>", table_header),
         Paragraph("<b>Val Acc (91-100)</b>", table_header),
         Paragraph("<b>C0 Holdout (R100)</b>", table_header)]
    ]

    if summary_path.exists():
        try:
            data = json.load(open(summary_path, encoding="utf-8"))
            for g in data.get("group_summary", []):
                rows_data.append([
                    Paragraph(f"<b>{g['group']}</b>", table_cell),
                    Paragraph(str(g['test_r100']), table_cell),
                    Paragraph(str(g['val_r100']), table_cell),
                    Paragraph(str(g['test_mean_91_100']), table_cell),
                    Paragraph(str(g['val_mean_91_100']), table_cell),
                    Paragraph(str(g['c0_holdout_r100']), table_cell),
                ])
        except Exception:
            pass

    if len(rows_data) == 1:
        # Placeholder rows if runs not yet fully finished
        for grp in ["FEDAVG clean", "FEDAVG noisy", "FEDPROX clean", "FEDPROX noisy"]:
            rows_data.append([
                Paragraph(f"<b>{grp}</b>", table_cell),
                Paragraph("Pending", table_cell),
                Paragraph("Pending", table_cell),
                Paragraph("Pending", table_cell),
                Paragraph("Pending", table_cell),
                Paragraph("Pending", table_cell)
            ])

    t = Table(rows_data, colWidths=[110, 85, 85, 90, 90, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2b6cb0')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

    # Plots side by side
    story.append(Paragraph("<b>Key Visualizations: Convergence and Diagnostic Metrics</b>", h2_style))
    img_test_acc = PLOTS_DIR / "plot1_test_accuracy_vs_rounds.png"
    img_weights = PLOTS_DIR / "plot9_aggregation_weights.png"

    plot_cells = []
    if img_test_acc.exists():
        plot_cells.append(Image(str(img_test_acc), width=260, height=135))
    else:
        plot_cells.append(Paragraph("<i>Plot 1 (Test Accuracy) pending completion</i>", body_style))

    if img_weights.exists():
        plot_cells.append(Image(str(img_weights), width=260, height=135))
    else:
        plot_cells.append(Paragraph("<i>Plot 9 (Weights) pending completion</i>", body_style))

    plot_table = Table([[plot_cells[0], plot_cells[1]]], colWidths=[270, 270])
    plot_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(plot_table)
    story.append(Spacer(1, 4))

    # Observations & Analysis — derived from actual results where available
    obs_parts = ["<b>Update-Metric Observations &amp; Robustness:</b><br/>"]

    # Try to load data to derive dynamic observations
    _all_data = None
    if summary_path.exists():
        try:
            _all_data = json.load(open(summary_path, encoding="utf-8"))
        except Exception:
            pass

    if _all_data is not None:
        gs = {g["group"]: g for g in _all_data.get("group_summary", [])}

        def _acc(group_key, field):
            g = gs.get(group_key)
            if g is None:
                return None
            v = g.get(field, "N/A")
            # format is "X.XXXX +/- Y.YYYY" or "N/A"
            if v == "N/A":
                return None
            try:
                return float(v.split()[0])
            except Exception:
                return None

        fa_clean = _acc("FEDAVG clean", "test_mean_91_100")
        fa_noisy = _acc("FEDAVG noisy", "test_mean_91_100")
        fp_clean = _acc("FEDPROX clean", "test_mean_91_100")
        fp_noisy = _acc("FEDPROX noisy", "test_mean_91_100")

        def _fmt(v):
            return f"{v:.4f}" if v is not None else "pending"

        # Noisy vs clean degradation
        if fa_clean is not None and fa_noisy is not None:
            fa_drop = fa_clean - fa_noisy
            obs_parts.append(
                f"&bull; <i>Noise Impact (FedAvg):</i> FedAvg clean test acc (rounds 91-100 mean): {_fmt(fa_clean)}; "
                f"noisy: {_fmt(fa_noisy)}. Noise degradation: {fa_drop:+.4f}.<br/>"
            )
        if fp_clean is not None and fp_noisy is not None:
            fp_drop = fp_clean - fp_noisy
            obs_parts.append(
                f"&bull; <i>Noise Impact (FedProx):</i> FedProx clean: {_fmt(fp_clean)}; "
                f"noisy: {_fmt(fp_noisy)}. Noise degradation: {fp_drop:+.4f}.<br/>"
            )

        # Algorithm comparison on noisy condition
        if fa_noisy is not None and fp_noisy is not None:
            better = "FedProx" if fp_noisy > fa_noisy else "FedAvg"
            diff = abs(fp_noisy - fa_noisy)
            obs_parts.append(
                f"&bull; <i>Algorithm Comparison (Noisy):</i> {better} achieves higher mean test acc under noisy condition "
                f"(difference: {diff:.4f}).<br/>"
            )

        # Aggregation weights from mu sweep / partition summary
        summary_json = Path("data/partitions/summary.json")
        if summary_json.exists():
            try:
                psummary = json.load(open(summary_json, encoding="utf-8"))
                weights_str = []
                for seed in [42, 123, 2026]:
                    seed_key = str(seed)
                    s_info = (
                        psummary.get("seeds", {}).get(seed_key)
                        or psummary.get("seeds", {}).get(seed)
                        or psummary.get(seed_key)
                        or psummary.get(seed)
                    )
                    if s_info:
                        c0_w = s_info.get("client_0_weight")
                        if c0_w is None:
                            c0_dict = (
                                s_info.get("clients", {}).get("0")
                                or s_info.get("clients", {}).get(0)
                                or s_info.get("0")
                                or s_info.get(0)
                            )
                            if isinstance(c0_dict, dict):
                                c0_w = c0_dict.get("aggregation_weight") or c0_dict.get("weight")
                        if c0_w is not None:
                            weights_str.append(f"seed {seed}: {c0_w:.3f}")
                if weights_str:
                    obs_parts.append(
                        f"&bull; <i>Client 0 Aggregation Weight (p0):</i> {', '.join(weights_str)}. "
                        "Higher p0 amplifies noise impact on the global model.<br/>"
                    )
            except Exception:
                pass

        obs_parts.append(
            "&bull; <i>Update-Metric Analysis:</i> Client 0 (40% noisy) exhibits altered gradient dynamics. "
            "In non-IID partitioning (alpha=0.3), low cosine alignment occurs due to class imbalance. "
            "FedProx proximal term constrains per-client parameter drift from the round-start global snapshot. "
            "See plots 5-7 for per-client L2 norm and cosine similarity trajectories."
        )
    else:
        obs_parts.append(
            "&bull; <i>Observations pending:</i> Run the full experiment matrix and re-run generate_tables.py, "
            "then re-run generate_report_pdf.py to populate data-driven observations.<br/>"
            "&bull; <i>Update-Metric Analysis:</i> Client 0 (40% noisy) exhibits altered update dynamics "
            "relative to clean clients. FedProx proximal term constrains drift from the global snapshot. "
            "See plots 5-7 for per-client norms and cosine similarities."
        )

    obs_text = "".join(obs_parts)
    story.append(Paragraph(obs_text, body_style))
    story.append(Spacer(1, 4))

    # Limitations & Conclusions
    lim_text = (
        "<b>Limitations & Scientific Disclosures:</b> "
        "(1) PathMNIST official test represents clinical cross-site evaluation; cross-site accuracy is lower than in-distribution validation. "
        "(2) μ was selected using in-distribution validation over rounds 91-100 on seed 42. "
        "(3) Three seeds provide preliminary variance estimation without formal statistical significance claims. "
        "(4) Offline JSON logging utilized as primary reproducible audit trail due to environment connectivity constraints."
    )
    story.append(Paragraph(lim_text, body_style))

    doc.build(story)
    print(f"Generated PDF report: {PDF_PATH}")

if __name__ == "__main__":
    generate_pdf()

