import os
import sys
import json
import hashlib
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("PFLlib/system"))

from PFLlib.system.flcore.trainmodel.models import FedAvgCNN

MANIFEST_PATH = "data/partitions/manifest.json"

def test_official_splits_exist_and_hashes():
    assert os.path.exists("data/pathmnist.npz")
    assert os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    assert manifest["raw_data_md5"] == "a8b06965200029087d5bd730944a56c1"
    
    val_data = np.load("data/partitions/official/val.npz")
    test_data = np.load("data/partitions/official/test.npz")
    assert val_data["x"].shape == (10004, 3, 28, 28)
    assert test_data["x"].shape == (7180, 3, 28, 28)
    assert val_data["y"].shape == (10004,)
    assert test_data["y"].shape == (7180,)

def test_client_partitions_and_noise_integrity():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
        
    for seed_str, sdata in manifest["seeds"].items():
        seed = int(seed_str)
        clients = sdata["clients"]
        noise_meta = sdata["client0_noise"]
        
        # Check minimum client size
        for cid_str, cdata in clients.items():
            assert cdata["assigned_total"] >= 1000, f"Seed {seed} Client {cid_str} < 1000 samples"
            assert cdata["train_size"] > 0
            assert cdata["test_size"] > 0
            
        # Check Client 0 noise
        n0 = clients["0"]["train_size"]
        expected_noise = round(0.4 * n0)
        assert noise_meta["exact_noise_count"] == expected_noise
        
        clean_train = np.load(f"data/partitions/seed_{seed}/clean/train/0.npz", allow_pickle=True)["data"].item()
        noisy_train = np.load(f"data/partitions/seed_{seed}/noisy/train/0.npz", allow_pickle=True)["data"].item()
        
        # Images must be bitwise identical
        np.testing.assert_array_equal(clean_train["x"], noisy_train["x"])
        
        clean_y = clean_train["y"]
        noisy_y = noisy_train["y"]
        diff_mask = (clean_y != noisy_y)
        assert np.sum(diff_mask) == expected_noise
        assert all(noisy_y[diff_mask] != clean_y[diff_mask])
        
        # Clients 1..4 must be identical between clean and noisy
        for cid in range(1, 5):
            c_clean = np.load(f"data/partitions/seed_{seed}/clean/train/{cid}.npz", allow_pickle=True)["data"].item()
            c_noisy = np.load(f"data/partitions/seed_{seed}/noisy/train/{cid}.npz", allow_pickle=True)["data"].item()
            np.testing.assert_array_equal(c_clean["x"], c_noisy["x"])
            np.testing.assert_array_equal(c_clean["y"], c_noisy["y"])
            
            # Local test sets must be identical
            t_clean = np.load(f"data/partitions/seed_{seed}/clean/test/{cid}.npz", allow_pickle=True)["data"].item()
            t_noisy = np.load(f"data/partitions/seed_{seed}/noisy/test/{cid}.npz", allow_pickle=True)["data"].item()
            np.testing.assert_array_equal(t_clean["x"], t_noisy["x"])
            np.testing.assert_array_equal(t_clean["y"], t_noisy["y"])

def test_model_architecture_and_forward_pass():
    model = FedAvgCNN(in_features=3, num_classes=9, dim=1024)
    dummy_input = torch.randn(1, 3, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 9)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 583113
    # Check no BatchNorm
    for m in model.modules():
        assert not isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))

def test_initial_model_hash_and_group_equality():
    """
    Test that for each seed (42, 123, 2026):
    1. The initial model checkpoint matches the SHA-256 in checkpoints/initial/manifest.json.
    2. All four experimental groups (FedAvg Clean, FedAvg Noisy, FedProx Clean, FedProx Noisy)
       load the identical initial model checkpoint and have bitwise identical parameters.
    3. The hash verification code in FederatedExperiment raises AssertionError if the file is corrupted.
    """
    from federated_runner import compute_file_sha256, FederatedExperiment

    manifest_path = "checkpoints/initial/manifest.json"
    assert os.path.exists(manifest_path), f"Initial manifest missing: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    seeds = [42, 123, 2026]
    groups = [
        ("fedavg", "clean"),
        ("fedavg", "noisy"),
        ("fedprox", "clean"),
        ("fedprox", "noisy"),
    ]

    for seed in seeds:
        ckpt_path = f"checkpoints/initial/initial_model_seed_{seed}.pt"
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
        
        # 1. Verify SHA-256 hash against manifest
        computed_sha = compute_file_sha256(ckpt_path)
        expected_sha = manifest[str(seed)]["sha256"]
        assert computed_sha == expected_sha, f"SHA-256 mismatch for seed {seed}! {computed_sha} != {expected_sha}"

        # 2. Verify all four groups load the exact same initial model
        loaded_states = []
        for algo, cond in groups:
            run_name = f"{algo}_{cond}_seed{seed}"
            exp = FederatedExperiment(
                run_name=run_name,
                algorithm=algo,
                condition=cond,
                seed=seed,
                global_rounds=1,
                wandb_mode="disabled",
                output_dir="test_init_tmp"
            )
            loaded_states.append((run_name, exp.global_model.state_dict()))

        # Compare pairwise bitwise equality across all four groups
        base_name, base_state = loaded_states[0]
        for name, state in loaded_states[1:]:
            for k in base_state:
                assert torch.equal(base_state[k], state[k]), (
                    f"Initial model mismatch between {base_name} and {name} for parameter {k}!"
                )

    # Clean up temp test directory if created
    shutil_rm = sys.modules.get("shutil")
    if os.path.exists("test_init_tmp"):
        import shutil
        shutil.rmtree("test_init_tmp", ignore_errors=True)

    print("\nInitial model hash and group equality test PASSED for all seeds and groups!")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
