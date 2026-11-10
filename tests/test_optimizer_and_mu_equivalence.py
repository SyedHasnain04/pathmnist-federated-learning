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

def test_mu_zero_equivalence_cuda_tolerant():
    """
    CUDA-tolerant version of the mu=0 equivalence test.
    
    Empirical Hardware Calibration & Diagnostic Numbers:
    ----------------------------------------------------
    On GPU (e.g., NVIDIA T4 on Google Colab), FP32 parallel-reduction operations in cuDNN
    convolution backward kernels are non-associative. Running the exact same FedAvg sequence
    twice sequentially from identical initial weights and batches produces a run-to-run
    divergence (hardware noise floor) of ~1.09e-4 after 15 steps (3 epochs x 5 batches)
    with SGD momentum=0.9, compounding exponentially over iterations.
    
    1-step variant:
      Across 1 single step (1 batch), non-associative accumulation has not compounded,
      yielding a tight parameter difference of <= 1e-6 (empirically ~1.0e-7 to 5.0e-7 on T4).
      
    Multi-step variant:
      Rather than an arbitrary fixed threshold (such as 1e-5 which fails when noise floor is ~1.09e-4),
      the multi-step test calibrates the intrinsic GPU noise floor by running FedAvg twice,
      and verifies that FedProx(mu=0) matches FedAvg within 3x the measured noise floor
      (diff_prox <= max(3.0 * noise_floor, 2e-4) on CUDA, and < 1e-7 on CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(100)
    init_model = FedAvgCNN(in_features=3, num_classes=9, dim=1024).to(device)
    
    batches = []
    for _ in range(5):
        bx = torch.randn(32, 3, 28, 28, device=device)
        by = torch.randint(0, 9, (32,), device=device)
        batches.append((bx, by))
        
    # --- Part 1: 1-step equivalence test (minimal FLOP roundoff accumulation) ---
    m_1step_avg = copy.deepcopy(init_model)
    run_local_training(m_1step_avg, batches[:1], lr=0.01, momentum=0.9, mu=0.0, global_snapshot=None, epochs=1)
    
    m_1step_prox = copy.deepcopy(init_model)
    snap_1step = [p.detach().clone() for p in m_1step_prox.parameters()]
    run_local_training(m_1step_prox, batches[:1], lr=0.01, momentum=0.9, mu=0.0, global_snapshot=snap_1step, epochs=1)
    
    diff_1step = max(torch.max(torch.abs(pa - pp)).item() for pa, pp in zip(m_1step_avg.parameters(), m_1step_prox.parameters()))
    tol_1step = 1e-6 if device.type == "cuda" else 1e-7
    assert diff_1step <= tol_1step, f"1-step mu=0 equivalence failed on {device}! diff: {diff_1step:.2e} > {tol_1step:.0e}"
    print(f"\n1-step mu=0 equivalence on {device}: diff = {diff_1step:.2e} <= {tol_1step:.0e}")
    
    # --- Part 2: Multi-step equivalence test relative to empirical noise floor ---
    # Run FedAvg twice from identical init and batches to measure hardware noise floor
    m_avg1 = copy.deepcopy(init_model)
    run_local_training(m_avg1, batches, lr=0.01, momentum=0.9, mu=0.0, global_snapshot=None, epochs=3)
    
    m_avg2 = copy.deepcopy(init_model)
    run_local_training(m_avg2, batches, lr=0.01, momentum=0.9, mu=0.0, global_snapshot=None, epochs=3)
    
    noise_floor = max(torch.max(torch.abs(p1 - p2)).item() for p1, p2 in zip(m_avg1.parameters(), m_avg2.parameters()))
    
    # Run FedProx(mu=0.0) from identical init and batches
    m_prox = copy.deepcopy(init_model)
    snapshot = [p.detach().clone() for p in m_prox.parameters()]
    run_local_training(m_prox, batches, lr=0.01, momentum=0.9, mu=0.0, global_snapshot=snapshot, epochs=3)
    
    diff_prox = max(torch.max(torch.abs(pa - pp)).item() for pa, pp in zip(m_avg1.parameters(), m_prox.parameters()))
    
    if device.type == "cuda":
        allowed_tol = max(3.0 * noise_floor, 2e-4)
        assert diff_prox <= allowed_tol, (
            f"CUDA mu=0 equivalence exceeded hardware noise floor on {device}! "
            f"diff: {diff_prox:.2e}, noise floor: {noise_floor:.2e}, allowed: {allowed_tol:.2e}"
        )
        print(f"Multi-step mu=0 equivalence on {device}: diff = {diff_prox:.2e}, noise floor = {noise_floor:.2e} (allowed: {allowed_tol:.2e})")
    else:
        assert diff_prox < 1e-7, f"CPU mu=0 equivalence failed! diff: {diff_prox:.2e} >= 1e-7"
        print(f"Multi-step mu=0 equivalence on {device}: diff = {diff_prox:.2e} < 1e-7 (noise floor = {noise_floor:.2e})")

def run_cuda_diagnostic():
    """
    Diagnostic function to run on CUDA / Colab.
    Evaluates:
      (a) Default cuDNN FedAvg run-to-run noise floor vs FedProx(mu=0)
      (b) 1-step FedAvg vs FedProx(mu=0) divergence
      (c) Deterministic flags impact on noise floor and run-to-run diff
      (d) Per-round timing impact with and without deterministic flags
    """
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}")
    print(f"CUDA DIAGNOSTIC & DETERMINISM NOISE FLOOR ANALYSIS")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"PyTorch version: {torch.__version__} | cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
    print(f"{'='*75}\n")
    
    torch.manual_seed(100)
    init_model = FedAvgCNN(in_features=3, num_classes=9, dim=1024).to(device)
    batches = [(torch.randn(32, 3, 28, 28, device=device), torch.randint(0, 9, (32,), device=device)) for _ in range(5)]
    
    # 1. 1-step test
    m_a1 = copy.deepcopy(init_model)
    run_local_training(m_a1, batches[:1], lr=0.01, momentum=0.9, mu=0.0, epochs=1)
    m_p1 = copy.deepcopy(init_model)
    snap1 = [p.detach().clone() for p in m_p1.parameters()]
    run_local_training(m_p1, batches[:1], lr=0.01, momentum=0.9, mu=0.0, global_snapshot=snap1, epochs=1)
    diff_1step = max(torch.max(torch.abs(p1 - p2)).item() for p1, p2 in zip(m_a1.parameters(), m_p1.parameters()))
    print(f"[1] 1-Step Equivalence (FedAvg vs FedProx mu=0):")
    print(f"    Max parameter difference: {diff_1step:.4e}")
    
    # 2. Standard (Default non-deterministic cuDNN)
    m_def_avg1 = copy.deepcopy(init_model)
    run_local_training(m_def_avg1, batches, lr=0.01, momentum=0.9, mu=0.0, epochs=3)
    m_def_avg2 = copy.deepcopy(init_model)
    run_local_training(m_def_avg2, batches, lr=0.01, momentum=0.9, mu=0.0, epochs=3)
    def_noise_floor = max(torch.max(torch.abs(p1 - p2)).item() for p1, p2 in zip(m_def_avg1.parameters(), m_def_avg2.parameters()))
    
    m_def_prox = copy.deepcopy(init_model)
    snap_def = [p.detach().clone() for p in m_def_prox.parameters()]
    run_local_training(m_def_prox, batches, lr=0.01, momentum=0.9, mu=0.0, global_snapshot=snap_def, epochs=3)
    def_diff_prox = max(torch.max(torch.abs(p1 - p2)).item() for p1, p2 in zip(m_def_avg1.parameters(), m_def_prox.parameters()))
    print(f"\n[2] Multi-Step Default (3 epochs, 15 steps):")
    print(f"    FedAvg run-to-run noise floor: {def_noise_floor:.4e}")
    print(f"    FedAvg vs FedProx(mu=0) diff:  {def_diff_prox:.4e}")
    print(f"    Ratio (diff / noise floor):    {(def_diff_prox / def_noise_floor if def_noise_floor > 0 else 1.0):.2f}x")
    
    # 3. Deterministic flags enabled
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"    Notice: use_deterministic_algorithms warning/exception: {e}")
        
    m_det_avg1 = copy.deepcopy(init_model)
    run_local_training(m_det_avg1, batches, lr=0.01, momentum=0.9, mu=0.0, epochs=3)
    m_det_avg2 = copy.deepcopy(init_model)
    run_local_training(m_det_avg2, batches, lr=0.01, momentum=0.9, mu=0.0, epochs=3)
    det_noise_floor = max(torch.max(torch.abs(p1 - p2)).item() for p1, p2 in zip(m_det_avg1.parameters(), m_det_avg2.parameters()))
    
    m_det_prox = copy.deepcopy(init_model)
    snap_det = [p.detach().clone() for p in m_det_prox.parameters()]
    run_local_training(m_det_prox, batches, lr=0.01, momentum=0.9, mu=0.0, global_snapshot=snap_det, epochs=3)
    det_diff_prox = max(torch.max(torch.abs(p1 - p2)).item() for p1, p2 in zip(m_det_avg1.parameters(), m_det_prox.parameters()))
    print(f"\n[3] Multi-Step with Deterministic Flags (cudnn.deterministic=True, CUBLAS_WORKSPACE_CONFIG=:4096:8):")
    print(f"    FedAvg run-to-run noise floor: {det_noise_floor:.4e}")
    print(f"    FedAvg vs FedProx(mu=0) diff:  {det_diff_prox:.4e}")
    
    # Reset deterministic flags
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
        
    # 4. Timing benchmark (1 round = 5 epochs on 20 batches of size 64 = 1,280 samples)
    bench_batches = [(torch.randn(64, 3, 28, 28, device=device), torch.randint(0, 9, (64,), device=device)) for _ in range(20)]
    
    # Warmup
    m_bench = copy.deepcopy(init_model)
    run_local_training(m_bench, bench_batches[:2], epochs=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    # Timing default
    t0 = time.time()
    for _ in range(3):
        m_b = copy.deepcopy(init_model)
        run_local_training(m_b, bench_batches, epochs=5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    t_default_per_round = (time.time() - t0) / 3.0
    
    # Timing deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    t0 = time.time()
    for _ in range(3):
        m_b = copy.deepcopy(init_model)
        run_local_training(m_b, bench_batches, epochs=5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    t_det_per_round = (time.time() - t0) / 3.0
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
        
    print(f"\n[4] Local Training Throughput Benchmark (per-client round equivalent):")
    print(f"    Default (cuDNN benchmark=True):      {t_default_per_round:.4f} s/round")
    print(f"    Deterministic (cudnn.deterministic): {t_det_per_round:.4f} s/round")
    slowdown = ((t_det_per_round - t_default_per_round) / t_default_per_round) * 100 if t_default_per_round > 0 else 0
    print(f"    Throughput overhead of determinism:   {slowdown:+.2f}%")
    print(f"{'='*75}\n")

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
    if "--diagnose" in sys.argv:
        run_cuda_diagnostic()
    else:
        pytest.main([__file__, "-v", "-s"])

