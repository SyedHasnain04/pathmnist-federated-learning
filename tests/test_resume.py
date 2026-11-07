"""
tests/test_resume.py - Test resume functionality and equivalence with uninterrupted runs.
Verifies that resuming from an intermediate checkpoint restores:
- Global model state
- Round counter
- Full experiment history
- RNG states (torch, cuda, numpy)
And produces identical results within floating-point tolerance (< 1e-6).
"""
import os
import sys
import copy
import shutil
import json
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("PFLlib/system"))

from federated_runner import FederatedExperiment, FederatedClient
from PFLlib.system.flcore.trainmodel.models import FedAvgCNN

class MiniExperiment(FederatedExperiment):
    """Subclass of FederatedExperiment using mini synthetic clients for fast deterministic testing."""
    def __init__(self, run_name, out_dir, global_rounds=2, checkpoint_interval=1):
        self.run_name = run_name
        self.algorithm = "fedavg"
        self.condition = "clean"
        self.seed = 42
        self.mu = 0.0
        self.global_rounds = global_rounds
        self.local_epochs = 1
        self.batch_size = 16
        self.lr = 0.01
        self.momentum = 0.9
        self.checkpoint_interval = checkpoint_interval
        self.is_pilot = False
        self.output_dir = os.path.join(out_dir, run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device("cpu")
        
        self.initial_model_sha256 = "synthetic_test_hash"
        self.global_model = FedAvgCNN(in_features=3, num_classes=9, dim=1024)
        torch.manual_seed(999)
        for p in self.global_model.parameters():
            p.data.normal_(0, 0.02)
            
        self.clients = []
        for cid in range(2):
            torch.manual_seed(100 + cid)
            tx = torch.randn(32, 3, 28, 28)
            ty = torch.randint(0, 9, (32,))
            self.clients.append(FederatedClient(cid, tx, ty, tx[:8], ty[:8], batch_size=16, device=self.device))
        self.client_sample_counts = [32, 32]
        self.aggregation_weights = [0.5, 0.5]
        self.env_metadata = {"device": "cpu", "gpu_name": "CPU", "torch_version": torch.__version__}
        self.use_wandb = False
        
        class MockEvaluator:
            def evaluate_official_val(self, m): return 0.55, 1.20
            def evaluate_official_test(self, m): return 0.52, 1.25
            def compute_confusion_matrix(self, m): return np.zeros((9, 9))
        self.evaluator = MockEvaluator()

def test_resume_equivalence_and_state_restoration():
    """
    Test that resuming from round-1 checkpoint restores:
    - Global model weights
    - Round counter
    - History records
    - RNG state
    and produces results numerically identical to an uninterrupted run.
    """
    test_dir = "test_resume_workspace"
    shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Save fixed base model
        base_init = FedAvgCNN(in_features=3, num_classes=9, dim=1024)
        init_path = os.path.join(test_dir, "init.pt")
        torch.save(base_init.state_dict(), init_path)
        
        # 1. Uninterrupted run: run 2 full rounds
        torch.manual_seed(42)
        np.random.seed(42)
        exp_uninterrupted = MiniExperiment("uninterrupted", test_dir, global_rounds=2, checkpoint_interval=1)
        exp_uninterrupted.global_model.load_state_dict(torch.load(init_path, weights_only=True))
        hist_unint = exp_uninterrupted.run()
        
        # 2. Resumed run (Phase 1): run 1 round only, saving checkpoint
        torch.manual_seed(42)
        np.random.seed(42)
        exp_resumed = MiniExperiment("resumed", test_dir, global_rounds=1, checkpoint_interval=1)
        exp_resumed.global_model.load_state_dict(torch.load(init_path, weights_only=True))
        hist_part1 = exp_resumed.run()
        
        # Verify checkpoint artifacts exist
        ckpt_model = os.path.join(exp_resumed.output_dir, "checkpoint_model.pt")
        ckpt_rng = os.path.join(exp_resumed.output_dir, "checkpoint_rng.pt")
        hist_file = os.path.join(exp_resumed.output_dir, "history.json")
        assert os.path.exists(ckpt_model), "checkpoint_model.pt must exist"
        assert os.path.exists(ckpt_rng), "checkpoint_rng.pt must exist"
        assert os.path.exists(hist_file), "history.json must exist"
        
        # Verify RNG checkpoint contents
        rng_payload = torch.load(ckpt_rng, map_location="cpu", weights_only=False)
        assert "torch_rng" in rng_payload
        assert "numpy_rng" in rng_payload
        assert rng_payload["round"] == 1
        
        # 3. Resumed run (Phase 2): instantiate and run to round 2 (should resume from round 1)
        # Disrupt RNG deliberately before resuming to ensure checkpoint restores state
        torch.manual_seed(99999)
        np.random.seed(99999)
        exp_resumed_cont = MiniExperiment("resumed", test_dir, global_rounds=2, checkpoint_interval=1)
        hist_resumed = exp_resumed_cont.run()
        
        # Verify round sequence and history integrity
        unint_rounds = [r["round"] for r in hist_unint["rounds"]]
        resumed_rounds = [r["round"] for r in hist_resumed["rounds"]]
        assert unint_rounds == [1, 2], f"Expected rounds [1, 2], got {unint_rounds}"
        assert resumed_rounds == [1, 2], f"Expected resumed rounds [1, 2], got {resumed_rounds}"
        
        # Verify model parameter equivalence between uninterrupted and resumed runs
        max_diff = 0.0
        for p_unint, p_res in zip(exp_uninterrupted.global_model.parameters(), exp_resumed_cont.global_model.parameters()):
            diff = torch.max(torch.abs(p_unint - p_res)).item()
            if diff > max_diff:
                max_diff = diff
        
        assert max_diff < 1e-6, f"Resumed model diverged from uninterrupted! Max diff: {max_diff}"
        
        # Verify metrics match between uninterrupted and resumed runs
        for r_idx in range(len(unint_rounds)):
            u_r = hist_unint["rounds"][r_idx]
            r_r = hist_resumed["rounds"][r_idx]
            assert abs(u_r["global_update_norm"] - r_r["global_update_norm"]) < 1e-5
            assert abs(u_r["client_train_losses"][0] - r_r["client_train_losses"][0]) < 1e-5
            
        print(f"\nResume equivalence test PASSED with max parameter diff: {max_diff:.2e} < 1e-6!")
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

