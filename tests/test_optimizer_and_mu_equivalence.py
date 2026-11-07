import os
import sys
import copy
import torch
import torch.nn as nn
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("PFLlib/system"))

from PFLlib.system.flcore.trainmodel.models import FedAvgCNN

def run_local_training(model, data_loader, lr=0.01, momentum=0.9, mu=0.0, global_snapshot=None, epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0)
    
    model.train()
    for _ in range(epochs):
        for x, y in data_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if mu > 0.0 and global_snapshot is not None:
                prox = 0.5 * mu * sum((p - g).pow(2).sum() for p, g in zip(model.parameters(), global_snapshot))
                loss = loss + prox
            loss.backward()
            optimizer.step()
    return model

def test_fresh_optimizer_momentum_isolation():
    torch.manual_seed(42)
    m = FedAvgCNN(in_features=3, num_classes=9, dim=1024)
    x = torch.randn(32, 3, 28, 28)
    y = torch.randint(0, 9, (32,))
    
    # Round 1
    opt1 = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    crit(m(x), y).backward()
    opt1.step()
    
    # State exists in opt1
    assert len(opt1.state) > 0
    param = next(m.parameters())
    assert "momentum_buffer" in opt1.state[param]
    
    # Round 2: fresh optimizer
    opt2 = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    assert len(opt2.state) == 0, "Fresh optimizer must have empty state"

def test_mu_zero_equivalence():
    torch.manual_seed(100)
    # Fixed initial model
    init_model = FedAvgCNN(in_features=3, num_classes=9, dim=1024)
    
    # Prepare identical synthetic data batches
    batches = []
    for _ in range(5):
        bx = torch.randn(32, 3, 28, 28)
        by = torch.randint(0, 9, (32,))
        batches.append((bx, by))
        
    # FedAvg
    model_fedavg = copy.deepcopy(init_model)
    run_local_training(model_fedavg, batches, lr=0.01, momentum=0.9, mu=0.0, global_snapshot=None, epochs=3)
    
    # FedProx with mu=0.0
    model_fedprox_zero = copy.deepcopy(init_model)
    snapshot = [p.detach().clone() for p in model_fedprox_zero.parameters()]
    run_local_training(model_fedprox_zero, batches, lr=0.01, momentum=0.9, mu=0.0, global_snapshot=snapshot, epochs=3)
    
    # Verify bitwise or tight floating-point equivalence
    for p_avg, p_prox in zip(model_fedavg.parameters(), model_fedprox_zero.parameters()):
        max_diff = torch.max(torch.abs(p_avg - p_prox)).item()
        assert max_diff < 1e-7, f"mu=0 equivalence failed! Max diff: {max_diff}"
    print("\nmu=0 equivalence passed with max diff < 1e-7!")

def test_mu_positive_restraining_effect():
    torch.manual_seed(200)
    init_model = FedAvgCNN(in_features=3, num_classes=9, dim=1024)
    batches = [(torch.randn(32, 3, 28, 28), torch.randint(0, 9, (32,))) for _ in range(5)]
    
    # FedAvg (mu=0)
    model_fedavg = copy.deepcopy(init_model)
    run_local_training(model_fedavg, batches, lr=0.01, momentum=0.9, mu=0.0, epochs=3)
    
    # FedProx (mu=0.1)
    model_fedprox = copy.deepcopy(init_model)
    snapshot = [p.detach().clone() for p in model_fedprox.parameters()]
    run_local_training(model_fedprox, batches, lr=0.01, momentum=0.9, mu=0.1, global_snapshot=snapshot, epochs=3)
    
    # Calculate distance from initialization w_t
    dist_fedavg = sum((p - g).pow(2).sum() for p, g in zip(model_fedavg.parameters(), snapshot)).sqrt().item()
    dist_fedprox = sum((p - g).pow(2).sum() for p, g in zip(model_fedprox.parameters(), snapshot)).sqrt().item()
    
    print(f"\nDistance from snapshot: FedAvg={dist_fedavg:.6f}, FedProx(mu=0.1)={dist_fedprox:.6f}")
    assert dist_fedprox < dist_fedavg, "Proximal term did not constrain distance to global model!"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

