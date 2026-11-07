import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class MetricsEvaluator:
    def __init__(self, val_path="data/partitions/official/val.npz", test_path="data/partitions/official/test.npz", batch_size=128, device="cpu"):
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Load official validation
        val_raw = np.load(val_path)
        self.val_x = torch.from_numpy(val_raw["x"])
        self.val_y = torch.from_numpy(val_raw["y"])
        val_ds = TensorDataset(self.val_x, self.val_y)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Load official test
        test_raw = np.load(test_path)
        self.test_x = torch.from_numpy(test_raw["x"])
        self.test_y = torch.from_numpy(test_raw["y"])
        test_ds = TensorDataset(self.test_x, self.test_y)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
    def evaluate_model(self, model, data_loader):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * y.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
                
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return accuracy, avg_loss
        
    def evaluate_official_val(self, model):
        return self.evaluate_model(model, self.val_loader)
        
    def evaluate_official_test(self, model):
        return self.evaluate_model(model, self.test_loader)
        
    def compute_confusion_matrix(self, model, num_classes=9):
        model.eval()
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                outputs = model(x)
                preds = outputs.argmax(dim=1).cpu().numpy()
                targets = y.numpy()
                for p, t in zip(preds, targets):
                    cm[t, p] += 1
        return cm

    def get_test_predictions_and_targets(self, model):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                outputs = model(x)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(y.numpy().tolist())
        return all_preds, all_targets

def flatten_trainable_params(model):
    # Flatten trainable parameters only, preserving deterministic order
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])

def compute_update_metrics(client_models, global_model_start, sample_counts):
    """
    Computes:
    - Delta_k = w_k_local - w_global_start
    - ||Delta_k||_2
    - Delta_global = sum_k p_k Delta_k
    - cos(Delta_k, Delta_global)
    - Delta_-k = sum_{j != k} n_j Delta_j / sum_{j != k} n_j
    - cos(Delta_k, Delta_-k)
    - Zero-norm safe handling
    """
    w_start = flatten_trainable_params(global_model_start)
    num_clients = len(client_models)
    
    deltas = []
    norms = []
    
    for model in client_models:
        w_k = flatten_trainable_params(model)
        delta_k = w_k - w_start
        norm_k = torch.norm(delta_k, p=2).item()
        deltas.append(delta_k)
        norms.append(norm_k)
        
    tot_samples = sum(sample_counts)
    weights = [n / tot_samples for n in sample_counts]
    
    # Global aggregated update
    delta_global = sum(w * d for w, d in zip(weights, deltas))
    global_norm = torch.norm(delta_global, p=2).item()
    
    standard_cosines = []
    leave_one_out_cosines = []
    
    for k in range(num_clients):
        d_k = deltas[k]
        n_k = norms[k]
        
        # Standard cosine against delta_global
        if n_k > 1e-12 and global_norm > 1e-12:
            cos_std = (torch.dot(d_k, delta_global) / (n_k * global_norm)).item()
            # Clamp to [-1.0, 1.0] to prevent slight numerical overflow
            cos_std = max(min(cos_std, 1.0), -1.0)
        else:
            cos_std = float("nan")
        standard_cosines.append(cos_std)
        
        # Leave-one-out cosine
        other_samples = sum(sample_counts[j] for j in range(num_clients) if j != k)
        if other_samples > 0:
            delta_minus_k = sum(sample_counts[j] * deltas[j] for j in range(num_clients) if j != k) / other_samples
            minus_k_norm = torch.norm(delta_minus_k, p=2).item()
            if n_k > 1e-12 and minus_k_norm > 1e-12:
                cos_loo = (torch.dot(d_k, delta_minus_k) / (n_k * minus_k_norm)).item()
                cos_loo = max(min(cos_loo, 1.0), -1.0)
            else:
                cos_loo = float("nan")
        else:
            cos_loo = float("nan")
        leave_one_out_cosines.append(cos_loo)
        
    return {
        "norms": norms,
        "standard_cosines": standard_cosines,
        "leave_one_out_cosines": leave_one_out_cosines,
        "weights": weights,
        "sample_counts": sample_counts,
        "global_norm": global_norm
    }

