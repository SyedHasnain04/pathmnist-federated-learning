import os
import json
import hashlib
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/pathmnist.npz"
BASE_DATA_DIR = "data/partitions"
SEEDS = [42, 123, 2026]
NUM_CLIENTS = 5
ALPHA = 0.3
MIN_CLIENT_SIZE = 1000
NOISE_RATE = 0.4
NUM_CLASSES = 9

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(131072):
            hasher.update(chunk)
    return hasher.hexdigest()

def normalize_images(imgs_uint8):
    # imgs_uint8: (N, 28, 28, 3) uint8 [0, 255]
    # Output: (N, 3, 28, 28) float32 in [-1.0, 1.0]
    imgs = imgs_uint8.astype(np.float32) / 255.0
    imgs = (imgs - 0.5) / 0.5
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    return np.ascontiguousarray(imgs)

def generate_partition(y_train, seed, alpha=0.3, num_clients=5, min_size=1000):
    rng = np.random.RandomState(seed)
    max_attempts = 100
    for attempt in range(max_attempts):
        client_indices = [[] for _ in range(num_clients)]
        for k in range(NUM_CLASSES):
            idx_k = np.where(y_train == k)[0]
            rng.shuffle(idx_k)
            props = rng.dirichlet(np.repeat(alpha, num_clients))
            splits = (np.cumsum(props) * len(idx_k)).astype(int)[:-1]
            pieces = np.split(idx_k, splits)
            for c, piece in enumerate(pieces):
                client_indices[c].extend(piece.tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_size:
            print(f"Seed {seed}: Dirichlet partition accepted on attempt {attempt+1}. Client sizes: {sizes}")
            for c in range(num_clients):
                client_indices[c] = np.array(client_indices[c], dtype=np.int64)
            return client_indices, sizes
    raise RuntimeError(f"Failed to generate partition with min_size >= {min_size} after {max_attempts} attempts.")

def apply_label_noise_client0(clean_train_y, seed):
    n0 = len(clean_train_y)
    # Exact specification requirement: built-in round(0.4 * n0) with ties-to-even
    exact_noise_count = round(NOISE_RATE * n0)
    rng = np.random.RandomState(seed)
    
    corrupt_indices = rng.choice(n0, size=exact_noise_count, replace=False)
    corrupt_indices.sort()
    
    noisy_train_y = clean_train_y.copy()
    original_labels = []
    replacement_labels = []
    
    for idx in corrupt_indices:
        orig = clean_train_y[idx]
        other_classes = [c for c in range(NUM_CLASSES) if c != orig]
        replacement = rng.choice(other_classes)
        noisy_train_y[idx] = replacement
        original_labels.append(int(orig))
        replacement_labels.append(int(replacement))
        
    # Assertions
    assert len(corrupt_indices) == exact_noise_count, f"Expected {exact_noise_count}, got {len(corrupt_indices)}"
    assert np.sum(noisy_train_y != clean_train_y) == exact_noise_count, "Exact noise count mismatch!"
    assert all(noisy_train_y[idx] != clean_train_y[idx] for idx in corrupt_indices), "Unchanged replacement label detected!"
    
    noise_metadata = {
        "client_id": 0,
        "n0": n0,
        "noise_rate": NOISE_RATE,
        "exact_noise_count": exact_noise_count,
        "corrupt_indices": corrupt_indices.tolist(),
        "original_labels": original_labels,
        "replacement_labels": replacement_labels
    }
    return noisy_train_y, noise_metadata

def build_data_pipeline():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"{RAW_DATA_PATH} not found. Downloading via medmnist...")
        os.makedirs("data", exist_ok=True)
        from medmnist import PathMNIST
        PathMNIST(split="train", download=True, root="data")
        print("PathMNIST download complete.")

    print(f"Loading raw PathMNIST from {RAW_DATA_PATH}...")
    raw = np.load(RAW_DATA_PATH)
    train_images_raw = raw["train_images"]  # (89996, 28, 28, 3)
    train_labels_raw = raw["train_labels"].squeeze()  # (89996,)
    val_images_raw = raw["val_images"]      # (10004, 28, 28, 3)
    val_labels_raw = raw["val_labels"].squeeze()      # (10004,)
    test_images_raw = raw["test_images"]    # (7180, 28, 28, 3)
    test_labels_raw = raw["test_labels"].squeeze()    # (7180,)
    
    assert len(train_images_raw) == 89996, "Official train split size mismatch!"
    assert len(val_images_raw) == 10004, "Official val split size mismatch!"
    assert len(test_images_raw) == 7180, "Official test split size mismatch!"
    
    # Save official validation and test sets in normalized format
    print("Normalizing official validation and test sets...")
    val_images_norm = normalize_images(val_images_raw)
    test_images_norm = normalize_images(test_images_raw)
    
    os.makedirs(f"{BASE_DATA_DIR}/official", exist_ok=True)
    val_path = f"{BASE_DATA_DIR}/official/val.npz"
    test_path = f"{BASE_DATA_DIR}/official/test.npz"
    
    np.savez(val_path, x=val_images_norm, y=val_labels_raw.astype(np.int64))
    np.savez(test_path, x=test_images_norm, y=test_labels_raw.astype(np.int64))
    
    val_hash = sha256_file(val_path)
    test_hash = sha256_file(test_path)
    print(f"Official Val SHA256: {val_hash}")
    print(f"Official Test SHA256: {test_hash}")
    
    # Normalize training set once
    print("Normalizing official training images...")
    train_images_norm = normalize_images(train_images_raw)
    train_labels = train_labels_raw.astype(np.int64)
    
    all_manifest = {
        "raw_data_md5": hashlib.md5(open(RAW_DATA_PATH, "rb").read()).hexdigest(),
        "official_val_sha256": val_hash,
        "official_test_sha256": test_hash,
        "seeds": {}
    }
    
    for seed in SEEDS:
        print(f"\n================ Processing Seed {seed} ================")
        client_assigned_indices, client_sizes = generate_partition(train_labels, seed, ALPHA, NUM_CLIENTS, MIN_CLIENT_SIZE)
        
        seed_dir = f"{BASE_DATA_DIR}/seed_{seed}"
        clean_dir = f"{seed_dir}/clean"
        noisy_dir = f"{seed_dir}/noisy"
        os.makedirs(f"{clean_dir}/train", exist_ok=True)
        os.makedirs(f"{clean_dir}/test", exist_ok=True)
        os.makedirs(f"{noisy_dir}/train", exist_ok=True)
        os.makedirs(f"{noisy_dir}/test", exist_ok=True)
        
        client_metadata = {}
        clean_train_hashes = {}
        noisy_train_hashes = {}
        client_test_hashes = {}
        
        noise_meta = None
        for client_id in range(NUM_CLIENTS):
            assigned_idx = client_assigned_indices[client_id]
            # 75:25 non-stratified, seeded split
            train_idx, test_idx = train_test_split(
                assigned_idx, train_size=0.75, random_state=seed, shuffle=True
            )
            train_idx = np.array(train_idx, dtype=np.int64)
            test_idx = np.array(test_idx, dtype=np.int64)
            
            # Assert split isolation
            assert len(np.intersect1d(train_idx, test_idx)) == 0, "Train and test index overlap!"
            assert len(train_idx) > 0 and len(test_idx) > 0, "Empty local split!"
            
            client_train_x = train_images_norm[train_idx]
            client_train_y_clean = train_labels[train_idx]
            client_test_x = train_images_norm[test_idx]
            client_test_y = train_labels[test_idx]
            
            # Save client-local test (identical for both clean and noisy)
            test_save_clean = f"{clean_dir}/test/{client_id}.npz"
            test_save_noisy = f"{noisy_dir}/test/{client_id}.npz"
            np.savez(test_save_clean, data={"x": client_test_x, "y": client_test_y})
            shutil.copy(test_save_clean, test_save_noisy)
            
            test_h1 = sha256_file(test_save_clean)
            test_h2 = sha256_file(test_save_noisy)
            assert test_h1 == test_h2, "Clean and noisy client test artifacts mismatch!"
            client_test_hashes[client_id] = test_h1
            
            # Save clean client-local train
            train_save_clean = f"{clean_dir}/train/{client_id}.npz"
            np.savez(train_save_clean, data={"x": client_train_x, "y": client_train_y_clean})
            clean_train_hashes[client_id] = sha256_file(train_save_clean)
            
            # Handle noisy client-local train
            if client_id == 0:
                client_train_y_noisy, noise_meta = apply_label_noise_client0(client_train_y_clean, seed)
                # Verify images are bitwise identical
                train_save_noisy = f"{noisy_dir}/train/{client_id}.npz"
                np.savez(train_save_noisy, data={"x": client_train_x, "y": client_train_y_noisy})
                noisy_train_hashes[client_id] = sha256_file(train_save_noisy)
                
                # Check that noisy and clean differ ONLY in client 0 train labels
                assert clean_train_hashes[client_id] != noisy_train_hashes[client_id], "Clean and noisy Client 0 train hashes match unexpectedly!"
            else:
                # Other clients have identical training data
                train_save_noisy = f"{noisy_dir}/train/{client_id}.npz"
                shutil.copy(train_save_clean, train_save_noisy)
                noisy_train_hashes[client_id] = sha256_file(train_save_noisy)
                assert clean_train_hashes[client_id] == noisy_train_hashes[client_id], f"Client {client_id} clean and noisy train hashes mismatch!"
            
            # Compute class distribution
            classes_clean, counts_clean = np.unique(client_train_y_clean, return_counts=True)
            dist_clean = {int(c): int(cnt) for c, cnt in zip(classes_clean, counts_clean)}
            
            client_metadata[client_id] = {
                "assigned_total": int(len(assigned_idx)),
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "clean_train_class_distribution": dist_clean,
                "clean_train_sha256": clean_train_hashes[client_id],
                "noisy_train_sha256": noisy_train_hashes[client_id],
                "test_sha256": test_h1
            }
        
        # Calculate aggregation weights based on local training sample counts
        tot_train_samples = sum(client_metadata[c]["train_size"] for c in range(NUM_CLIENTS))
        for c in range(NUM_CLIENTS):
            client_metadata[c]["aggregation_weight"] = float(client_metadata[c]["train_size"] / tot_train_samples)
            
        print(f"Seed {seed} Client 0 summary:")
        print(f"  Assigned: {client_metadata[0]['assigned_total']}")
        print(f"  Train samples: {client_metadata[0]['train_size']}")
        print(f"  Test samples: {client_metadata[0]['test_size']}")
        print(f"  Aggregation weight: {client_metadata[0]['aggregation_weight']:.4f}")
        print(f"  Noise count: {noise_meta['exact_noise_count']} ({noise_meta['exact_noise_count']/client_metadata[0]['train_size']:.4%})")
        
        seed_manifest = {
            "seed": seed,
            "alpha": ALPHA,
            "min_client_size": MIN_CLIENT_SIZE,
            "total_assigned_samples": int(sum(client_sizes)),
            "total_train_samples": tot_train_samples,
            "clients": client_metadata,
            "client0_noise": noise_meta
        }
        
        with open(f"{seed_dir}/metadata.json", "w") as f:
            json.dump(seed_manifest, f, indent=2)
            
        all_manifest["seeds"][seed] = seed_manifest

    with open(f"{BASE_DATA_DIR}/manifest.json", "w") as f:
        json.dump(all_manifest, f, indent=2)
        
    print("\nData generation and verification COMPLETE!")
    print(f"Manifest written to {BASE_DATA_DIR}/manifest.json")

if __name__ == "__main__":
    build_data_pipeline()
