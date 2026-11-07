"""
package_for_colab.py - Build a clean, self-contained zip package for Google Colab.
Excludes large cache files, datasets (which auto-download in Colab), and local logs.
"""
import os
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
ZIP_PATH = BASE_DIR / "colab_package.zip"

INCLUDE_FILES = [
    "federated_runner.py",
    "metrics_evaluator.py",
    "orchestrator.py",
    "generate_pathmnist_data.py",
    "generate_initial_models.py",
    "generate_plots.py",
    "generate_tables.py",
    "generate_report_pdf.py",
    "PathMNIST_FedAvg_FedProx_Final_Execution_Plan.md",
    "PathMNIST_FedAvg_vs_FedProx_Colab.ipynb",
    "README.md",
    "DEVIATIONS.md",
    ".gitignore"
]

INCLUDE_DIRS = [
    "tests",
    "checkpoints/initial",
    "PFLlib/system/flcore/trainmodel"
]

def make_colab_zip():
    print(f"Creating {ZIP_PATH.name}...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add root files
        for fname in INCLUDE_FILES:
            fpath = BASE_DIR / fname
            if fpath.exists():
                zipf.write(fpath, arcname=fname)
                print(f"  Added {fname}")

        # Add directories
        for dname in INCLUDE_DIRS:
            dpath = BASE_DIR / dname
            if dpath.exists():
                for root, _, files in os.walk(dpath):
                    if "__pycache__" in root or ".pytest_cache" in root:
                        continue
                    for file in files:
                        fp = Path(root) / file
                        arc = fp.relative_to(BASE_DIR)
                        zipf.write(fp, arcname=str(arc))
                        print(f"  Added {arc}")

    size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"\nSuccessfully built {ZIP_PATH.name} ({size_mb:.2f} MB).")
    print("Ready to upload to Google Colab!")

if __name__ == "__main__":
    make_colab_zip()

