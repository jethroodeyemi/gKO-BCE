# dataset_normalizer.py
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer

import config


def _make_quantile_normalizer(X_train, seed=config.SEED, noise=config.QUANTILE_NOISE):
    """Build a QuantileTransformer (normal output) fitted on all training columns.

    Mirrors the transformation used to train the released KO-BCE / gKO-BCE weights:
    a small amount of Gaussian noise is added before fitting to de-duplicate ties.
    """
    normalizer = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
        subsample=int(1e9),
        random_state=seed,
    )
    X_fit = X_train
    if noise and noise > 0:
        stds = np.std(X_train, axis=0, keepdims=True)
        noise_std = noise / np.maximum(stds, noise)
        X_fit = X_train + noise_std * np.random.default_rng(seed).standard_normal(X_train.shape)
    normalizer.fit(X_fit)
    return normalizer


def normalize_and_save():
    """Read the structured .pkl dataset, apply a QuantileTransformer fitted on the
    training partition (ALL feature columns), and save .npy matrices + the fitted
    normalizer for training and inference."""
    print("--- Starting Tabular Dataset Normalization & Packaging ---")

    if not os.path.exists(config.STRUCTURED_DATA_PATH):
        raise FileNotFoundError(f"Source file not found at '{config.STRUCTURED_DATA_PATH}'. Run feature_extractor first.")
    if not os.path.exists(config.SPLITS_FILE_PATH):
        raise FileNotFoundError(f"Splits file not found at '{config.SPLITS_FILE_PATH}'. Run sequence_clustering first.")

    print(f"Loading structured data from: {config.STRUCTURED_DATA_PATH}")
    with open(config.STRUCTURED_DATA_PATH, "rb") as f:
        protein_data_list = pickle.load(f)

    print(f"Loading data splits from: {config.SPLITS_FILE_PATH}")
    with open(config.SPLITS_FILE_PATH, "r") as f:
        splits = json.load(f)

    # Reconstruct the flat residue-level arrays.
    features, labels, groups = [], [], []
    for protein_data in tqdm(protein_data_list, desc="Processing proteins"):
        features.append(protein_data["X_arr"])
        labels.append(protein_data["df_stats"]["is_epitope"].values)
        groups.append(np.full(protein_data["length"], protein_data["pdb_id"]))

    X = np.vstack(features).astype(np.float32)
    y = np.concatenate(labels)
    groups = np.concatenate(groups)
    print(f"Full dataset reconstructed. Residues: {len(y)}, Features: {X.shape[1]}")

    # Partition by pre-computed homology-aware splits.
    train_groups, val_groups, test_groups = splits["train"], splits["val"], splits["test"]
    train_mask = np.isin(groups, train_groups)
    val_mask = np.isin(groups, val_groups)
    test_mask = np.isin(groups, test_groups)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  - Train: {len(y_train)} residues / {len(train_groups)} proteins")
    print(f"  - Val:   {len(y_val)} residues / {len(val_groups)} proteins")
    print(f"  - Test:  {len(y_test)} residues / {len(test_groups)} proteins")

    # Fit the QuantileTransformer on TRAIN only, transform all splits.
    print("Fitting QuantileTransformer (output_distribution='normal') on all columns...")
    normalizer = _make_quantile_normalizer(X_train)
    X_train = normalizer.transform(X_train).astype(np.float32)
    X_val = normalizer.transform(X_val).astype(np.float32)
    X_test = normalizer.transform(X_test).astype(np.float32)

    os.makedirs(config.DATASTORE_DIR, exist_ok=True)
    np.save(os.path.join(config.DATASTORE_DIR, "X_num_train.npy"), X_train)
    np.save(os.path.join(config.DATASTORE_DIR, "X_num_val.npy"), X_val)
    np.save(os.path.join(config.DATASTORE_DIR, "X_num_test.npy"), X_test)
    np.save(os.path.join(config.DATASTORE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(config.DATASTORE_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(config.DATASTORE_DIR, "y_test.npy"), y_test)

    # Save the fitted normalizer (shipped alongside the checkpoint for inference).
    normalizer_path = os.path.join(config.DATASTORE_DIR, "normalizer_quantile.pkl")
    with open(normalizer_path, "wb") as f:
        pickle.dump(normalizer, f)
    print(f"Normalizer saved to: {normalizer_path}")

    info = {
        "name": "gko_bce_dataset",
        "task_type": "binclass",
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "n_num_features": int(X_train.shape[1]),
        "n_cat_features": 0,
        "n_classes": 1,
        "normalization": "quantile",
    }
    with open(os.path.join(config.DATASTORE_DIR, "info.json"), "w") as f:
        json.dump(info, f, indent=4)
    print("--- Tabular Dataset Packaging Complete ---")


if __name__ == "__main__":
    normalize_and_save()
