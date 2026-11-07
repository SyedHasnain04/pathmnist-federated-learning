import os
import random
import hashlib
import json
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("PFLlib/system"))

from PFLlib.system.flcore.trainmodel.models import FedAvgCNN

SEEDS = [42, 123, 2026]
MODEL_DIR = "checkpoints/initial"

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def create_initial_checkpoints():
    os.makedirs(MODEL_DIR, exist_ok=True)
    manifest = {}
    
    for seed in SEEDS:
        set_all_seeds(seed)
        model = FedAvgCNN(in_features=3, num_classes=9, dim=1024)
        path = os.path.join(MODEL_DIR, f"initial_model_seed_{seed}.pt")
        torch.save(model.state_dict(), path)
        model_hash = sha256_file(path)
        manifest[seed] = {
            "checkpoint_path": path,
            "sha256": model_hash
        }
        print(f"Seed {seed}: saved to {path}, SHA256: {model_hash}")
        
    with open(os.path.join(MODEL_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    create_initial_checkpoints()

