import os
import sys

# Limit MKL / OpenMP threads before importing torch/numpy
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "4"
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

import copy
import json
import time
import hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure local imports work
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("PFLlib/system"))

from PFLlib.system.flcore.trainmodel.models import FedAvgCNN
from metrics_evaluator import MetricsEvaluator, compute_update_metrics, flatten_trainable_params

def compute_file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(131072):
            hasher.update(chunk)
    return hasher.hexdigest()

class FederatedClient:
    def __init__(self, client_id, train_x, train_y, test_x, test_y, batch_size=64, device="cpu"):
        self.client_id = client_id
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # Keep tensors in memory
        self.train_x = train_x.clone()
        self.train_y = train_y.clone()
        self.test_x = test_x.clone()
        self.test_y = test_y.clone()
        
        self.train_samples = len(self.train_y)
        self.test_samples = len(self.test_y)
        
        # Datasets
        self.train_ds = TensorDataset(self.train_x, self.train_y)
        self.test_ds = TensorDataset(self.test_x, self.test_y)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()

    def train_round(self, base_model, local_epochs=5, lr=0.01, momentum=0.9, mu=0.0):
        local_model = copy.deepcopy(base_model).to(self.device)
        local_model.train()
        
        # FedProx: capture detached snapshot of start-of-round parameters
        if mu > 0.0:
            snapshot = [p.detach().clone() for p in local_model.parameters()]
        else:
            snapshot = None
            
        # Fresh optimizer every communication round - NO momentum carried across rounds
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0)
        
        # shuffle=True for each round
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        
        round_losses = []
        for epoch in range(local_epochs):
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                out = local_model(x)
                ce_loss = self.criterion(out, y)
                
                if mu > 0.0 and snapshot is not None:
                    prox_loss = 0.5 * mu * sum((p - g).pow(2).sum() for p, g in zip(local_model.parameters(), snapshot))
                    loss = ce_loss + prox_loss
                else:
                    loss = ce_loss
                    
                loss.backward()
                optimizer.step()
                round_losses.append(ce_loss.item())
                
        avg_train_loss = float(np.mean(round_losses)) if round_losses else 0.0
        return local_model, avg_train_loss

    def evaluate_holdout(self, model):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = model(x)
                loss = self.criterion(out, y)
                total_loss += loss.item() * y.size(0)
                preds = out.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
        acc = total_correct / total_samples if total_samples > 0 else 0.0
        loss = total_loss / total_samples if total_samples > 0 else 0.0
        return acc, loss

class FederatedExperiment:
    def __init__(self, run_name, algorithm, condition, seed, mu=0.0, global_rounds=100, local_epochs=5,
                 batch_size=64, lr=0.01, momentum=0.9, num_threads=8, device=None, checkpoint_interval=5,
                 wandb_project="pathmnist-federated-learning", wandb_mode="online", wandb_api_key=None,
                 output_dir="results"):
        self.run_name = run_name
        self.algorithm = algorithm.lower()
        self.condition = condition.lower()
        self.seed = int(seed)
        self.mu = float(mu) if self.algorithm == "fedprox" else 0.0
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = os.path.join(output_dir, self.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Device selection: use CUDA if available, unless specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[{self.run_name}] Running on compute device: {self.device}", flush=True)
        
        # Threading for CPU operations
        if self.device.type == "cpu":
            torch.set_num_threads(num_threads)
        
        # Load initial model checkpoint
        ckpt_path = f"checkpoints/initial/initial_model_seed_{self.seed}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Initial model checkpoint missing: {ckpt_path}")
        
        # Verify initial checkpoint SHA256 against manifest
        self.initial_model_sha256 = compute_file_sha256(ckpt_path)
        manifest_path = "checkpoints/initial/manifest.json"
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
                if str(self.seed) in manifest:
                    expected_hash = manifest[str(self.seed)]["sha256"]
                    assert self.initial_model_sha256 == expected_hash, (
                        f"Initial model hash mismatch for seed {self.seed}! "
                        f"Expected {expected_hash}, got {self.initial_model_sha256}"
                    )
                    print(f"Verified initial model SHA256 for seed {self.seed}: {self.initial_model_sha256}")

        self.global_model = FedAvgCNN(in_features=3, num_classes=9, dim=1024).to(self.device)
        self.global_model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        
        # Evaluator for official val and test
        self.evaluator = MetricsEvaluator(device=self.device)
        
        # Load client datasets
        partition_dir = f"data/partitions/seed_{self.seed}/{self.condition}"
        self.clients = []
        self.client_sample_counts = []
        for cid in range(5):
            train_raw = np.load(f"{partition_dir}/train/{cid}.npz", allow_pickle=True)["data"].item()
            test_raw = np.load(f"{partition_dir}/test/{cid}.npz", allow_pickle=True)["data"].item()
            
            tx = torch.from_numpy(train_raw["x"]).float()
            ty = torch.from_numpy(train_raw["y"]).long()
            te_x = torch.from_numpy(test_raw["x"]).float()
            te_y = torch.from_numpy(test_raw["y"]).long()
            
            client = FederatedClient(cid, tx, ty, te_x, te_y, batch_size=self.batch_size, device=self.device)
            self.clients.append(client)
            self.client_sample_counts.append(client.train_samples)
            
        tot_samples = sum(self.client_sample_counts)
        self.aggregation_weights = [n / tot_samples for n in self.client_sample_counts]

        # Environment metadata
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and self.device.type == "cuda" else "CPU"
        self.env_metadata = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device),
            "gpu_name": device_name,
            "cpu_count": os.cpu_count()
        }
        
        # Setup W&B (offline or online)
        self.use_wandb = False
        if wandb_project and wandb_mode != "disabled":
            try:
                import wandb
                # Key resolution: argument -> environment variable
                effective_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
                if effective_key:
                    wandb.login(key=effective_key)
                
                # Determine mode
                eff_mode = wandb_mode
                if eff_mode == "online" and not effective_key and not os.path.exists(os.path.expanduser("~/.netrc")):
                    print("W&B API key not detected. Falling back to 'offline' mode.")
                    eff_mode = "offline"

                wandb.init(
                    project=wandb_project,
                    name=self.run_name,
                    mode=eff_mode,
                    config={
                        "run_name": self.run_name,
                        "algorithm": self.algorithm,
                        "condition": self.condition,
                        "seed": self.seed,
                        "mu": self.mu,
                        "global_rounds": self.global_rounds,
                        "local_epochs": self.local_epochs,
                        "batch_size": self.batch_size,
                        "lr": self.lr,
                        "momentum": self.momentum,
                        "client_sample_counts": self.client_sample_counts,
                        "aggregation_weights": self.aggregation_weights,
                        "client_0_train_size": self.client_sample_counts[0],
                        "client_0_weight": self.aggregation_weights[0],
                        "initial_model_sha256": self.initial_model_sha256,
                        "env": self.env_metadata,
                        "architecture": "FedAvgCNN (3x28x28 -> 9)",
                        "num_clients": 5,
                        "join_ratio": 1.0
                    }
                )
                self.use_wandb = True
                self.wandb = wandb
                print(f"W&B logging active in mode='{eff_mode}' (Project: {wandb_project}, Run: {self.run_name})")
            except Exception as e:
                print(f"W&B initialization skipped or failed ({e}). Comprehensive local JSON logging will be used.")
        else:
            print("W&B disabled. Using comprehensive local JSON logging.")
            
    def aggregate_models(self, client_models):
        """
        w_global^{t+1} = sum_k p_k w_{k, local}^t
        """
        new_global = copy.deepcopy(self.global_model)
        for param in new_global.parameters():
            param.data.zero_()
            
        for p_k, client_m in zip(self.aggregation_weights, client_models):
            for g_param, c_param in zip(new_global.parameters(), client_m.parameters()):
                g_param.data += p_k * c_param.data.clone()
        return new_global

    def run(self):
        print(f"\n=======================================================", flush=True)
        print(f"Starting Run: {self.run_name}", flush=True)
        print(f"Algorithm: {self.algorithm.upper()} | Condition: {self.condition.upper()} | Seed: {self.seed} | Mu: {self.mu}", flush=True)
        print(f"Client train counts: {self.client_sample_counts}", flush=True)
        print(f"Aggregation weights: {[round(w, 4) for w in self.aggregation_weights]}", flush=True)
        print(f"Initial model SHA256: {self.initial_model_sha256}", flush=True)
        print(f"Compute Environment: {self.env_metadata}", flush=True)
        print(f"=======================================================", flush=True)
        
        # --- Resume logic: check for existing partial history & RNG state ---
        history_path = os.path.join(self.output_dir, "history.json")
        checkpoint_model_path = os.path.join(self.output_dir, "checkpoint_model.pt")
        checkpoint_rng_path = os.path.join(self.output_dir, "checkpoint_rng.pt")
        resume_round = 0
        history = {
            "config": {
                "run_name": self.run_name,
                "algorithm": self.algorithm,
                "condition": self.condition,
                "seed": self.seed,
                "mu": self.mu,
                "global_rounds": self.global_rounds,
                "local_epochs": self.local_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "momentum": self.momentum,
                "client_sample_counts": self.client_sample_counts,
                "aggregation_weights": self.aggregation_weights,
                "initial_model_sha256": self.initial_model_sha256,
                "env": self.env_metadata
            },
            "rounds": []
        }
        
        if os.path.exists(history_path) and os.path.exists(checkpoint_model_path):
            try:
                existing = json.load(open(history_path, encoding="utf-8"))
                completed_rounds = existing.get("rounds", [])
                if completed_rounds:
                    resume_round = completed_rounds[-1]["round"]
                    if resume_round >= self.global_rounds:
                        print(f"Run already complete ({resume_round} rounds). Skipping.", flush=True)
                        return existing
                    # Restore model state
                    self.global_model.load_state_dict(torch.load(checkpoint_model_path, map_location=self.device, weights_only=True))
                    # Restore RNG state if available
                    if os.path.exists(checkpoint_rng_path):
                        rng_state = torch.load(checkpoint_rng_path, map_location="cpu", weights_only=False)
                        torch.set_rng_state(rng_state["torch_rng"])
                        if torch.cuda.is_available() and rng_state.get("cuda_rng") is not None:
                            torch.cuda.set_rng_state_all(rng_state["cuda_rng"])
                        if "numpy_rng" in rng_state:
                            np.random.set_state(rng_state["numpy_rng"])
                        print(f"Restored RNG state from checkpoint at round {resume_round}.", flush=True)
                    history["rounds"] = completed_rounds
                    print(f"RESUMING from round {resume_round}. {self.global_rounds - resume_round} rounds remaining.", flush=True)
            except Exception as e:
                print(f"WARNING: Could not resume ({e}). Starting from scratch.", flush=True)
                resume_round = 0
                history["rounds"] = []
        
        start_time_all = time.time()
        
        if resume_round == 0:
            # Round 0 evaluation (before training)
            print("Evaluating Round 0 initial model...", flush=True)
            val_acc, val_loss = self.evaluator.evaluate_official_val(self.global_model)
            test_acc, test_loss = self.evaluator.evaluate_official_test(self.global_model)
            print(f"Round 0 (Initial): Official Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}", flush=True)
        
        for r in range(resume_round + 1, self.global_rounds + 1):
            r_start_t = time.time()
            
            # 1. Snapshot global model at round start
            global_model_start = copy.deepcopy(self.global_model)
            
            # 2. Local client training
            client_models = []
            client_train_losses = []
            for client in self.clients:
                c_start_t = time.time()
                local_m, loss_c = client.train_round(
                    base_model=global_model_start,
                    local_epochs=self.local_epochs,
                    lr=self.lr,
                    momentum=self.momentum,
                    mu=self.mu
                )
                client_models.append(local_m)
                client_train_losses.append(loss_c)
                print(f"  [Round {r}] Client {client.client_id} ({client.train_samples} samples) trained in {time.time()-c_start_t:.2f}s | Loss: {loss_c:.4f}", flush=True)
                
            # 3. Capture update metrics BEFORE aggregation
            update_stats = compute_update_metrics(
                client_models=client_models,
                global_model_start=global_model_start,
                sample_counts=self.client_sample_counts
            )
            
            # 4. Aggregate models
            self.global_model = self.aggregate_models(client_models)
            
            # 5. Evaluate on official validation and official global test
            val_acc, val_loss = self.evaluator.evaluate_official_val(self.global_model)
            test_acc, test_loss = self.evaluator.evaluate_official_test(self.global_model)
            
            # 6. Evaluate on per-client local holdouts
            client_holdout_accs = []
            client_holdout_losses = []
            for client in self.clients:
                c_acc, c_loss = client.evaluate_holdout(self.global_model)
                client_holdout_accs.append(c_acc)
                client_holdout_losses.append(c_loss)
                
            r_duration = time.time() - r_start_t
            
            round_record = {
                "round": r,
                "duration_seconds": r_duration,
                "official_val_accuracy": val_acc,
                "official_val_loss": val_loss,
                "official_test_accuracy": test_acc,
                "official_test_loss": test_loss,
                "client_train_losses": client_train_losses,
                "client_holdout_accuracies": client_holdout_accs,
                "client_holdout_losses": client_holdout_losses,
                "update_norms": update_stats["norms"],
                "standard_cosines": update_stats["standard_cosines"],
                "leave_one_out_cosines": update_stats["leave_one_out_cosines"],
                "global_update_norm": update_stats["global_norm"]
            }
            history["rounds"].append(round_record)
            
            # Log to W&B
            if self.use_wandb:
                log_dict = {
                    "round": r,
                    "official_val/accuracy": val_acc,
                    "official_val/loss": val_loss,
                    "official_test/accuracy": test_acc,
                    "official_test/loss": test_loss,
                    "global_update_norm": update_stats["global_norm"],
                    "round_duration_seconds": r_duration
                }
                for cid in range(5):
                    log_dict[f"client_{cid}/holdout_acc"] = client_holdout_accs[cid]
                    log_dict[f"client_{cid}/holdout_loss"] = client_holdout_losses[cid]
                    log_dict[f"client_{cid}/update_norm"] = update_stats["norms"][cid]
                    log_dict[f"client_{cid}/train_loss"] = client_train_losses[cid]
                    if not np.isnan(update_stats["standard_cosines"][cid]):
                        log_dict[f"client_{cid}/standard_cosine"] = update_stats["standard_cosines"][cid]
                    if not np.isnan(update_stats["leave_one_out_cosines"][cid]):
                        log_dict[f"client_{cid}/loo_cosine"] = update_stats["leave_one_out_cosines"][cid]
                self.wandb.log(log_dict)
                
            if r % 10 == 0 or r == self.global_rounds or r == 1:
                print(f"Round {r:3d}/{self.global_rounds} | Time: {r_duration:.1f}s | "
                      f"Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | "
                      f"C0 Norm: {update_stats['norms'][0]:.4f} | C0 StdCos: {update_stats['standard_cosines'][0]:.4f}")
                
            # Intermediate checkpoint every checkpoint_interval rounds
            if r % self.checkpoint_interval == 0:
                with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
                torch.save(self.global_model.state_dict(), os.path.join(self.output_dir, "checkpoint_model.pt"))
                rng_state = {
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy_rng": np.random.get_state(),
                    "round": r
                }
                torch.save(rng_state, os.path.join(self.output_dir, "checkpoint_rng.pt"))
                    
        total_time = time.time() - start_time_all
        print(f"\nCompleted {self.global_rounds} rounds in {total_time/60:.2f} minutes.")
        
        # Final evaluation: compute confusion matrix on official test
        cm = self.evaluator.compute_confusion_matrix(self.global_model)
        np.save(os.path.join(self.output_dir, "confusion_matrix.npy"), cm)
        
        # W&B final confusion matrix logging
        if self.use_wandb:
            try:
                preds, targets = self.evaluator.get_test_predictions_and_targets(self.global_model)
                class_names = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
                self.wandb.log({
                    "test/confusion_matrix": self.wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=targets,
                        preds=preds,
                        class_names=class_names
                    )
                })
            except Exception as e:
                print(f"Notice: W&B confusion matrix visualization skipped ({e}).")

        # Save final model checkpoint and full history
        torch.save(self.global_model.state_dict(), os.path.join(self.output_dir, "final_model.pt"))
        with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
            
        if self.use_wandb:
            self.wandb.finish()
            
        return history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default="fedavg", choices=["fedavg", "fedprox"])
    parser.add_argument("--condition", type=str, default="clean", choices=["clean", "noisy"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu'). Defaults to cuda if available.")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Rounds between resume checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory where results are stored")
    parser.add_argument("--wandb_project", type=str, default="pathmnist-federated-learning")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_api_key", type=str, default=None)
    args = parser.parse_args()
    
    runner = FederatedExperiment(
        run_name=args.run_name,
        algorithm=args.algorithm,
        condition=args.condition,
        seed=args.seed,
        mu=args.mu,
        global_rounds=args.rounds,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        num_threads=args.threads,
        device=args.device,
        checkpoint_interval=args.checkpoint_interval,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        wandb_api_key=args.wandb_api_key,
        output_dir=args.results_dir
    )
    runner.run()
