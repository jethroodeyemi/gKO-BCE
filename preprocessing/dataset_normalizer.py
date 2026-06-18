# dataset_normalizer.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import config

def normalize_and_save():
    """
    Reads the structured .pkl dataset, normalizes continuous features using StandardScaler fitted
    exclusively on the training partition, and saves separate .npy files and metadata for model training.
    """
    print("--- Starting Tabular Dataset Normalization & Packaging ---")

    # 1. Check for source files
    if not os.path.exists(config.STRUCTURED_DATA_PATH):
        raise FileNotFoundError(f"Source file not found at '{config.STRUCTURED_DATA_PATH}'. Run feature_extractor first.")
    if not os.path.exists(config.SPLITS_FILE_PATH):
        raise FileNotFoundError(f"Splits file not found at '{config.SPLITS_FILE_PATH}'. Run sequence_clustering first.")

    # 2. Load the source data and splits
    print(f"Loading structured data from: {config.STRUCTURED_DATA_PATH}")
    with open(config.STRUCTURED_DATA_PATH, 'rb') as f:
        protein_data_list = pickle.load(f)

    print(f"Loading data splits from: {config.SPLITS_FILE_PATH}")
    with open(config.SPLITS_FILE_PATH, 'r') as f:
        splits = json.load(f)

    # 3. Reconstruct the full, flat dataset arrays
    features, labels, groups = [], [], []
    for protein_data in tqdm(protein_data_list, desc="Processing proteins"):
        features.append(protein_data['X_arr'])
        labels.append(protein_data['df_stats']['is_epitope'].values)
        groups.append(np.full(protein_data['length'], protein_data['pdb_id']))
        
    X = np.vstack(features)
    y = np.concatenate(labels)
    groups = np.concatenate(groups)
    print(f"Full dataset reconstructed. Total residues: {len(y)}, Features: {X.shape[1]}")

    # 4. Partition the data based on the pre-computed splits
    train_groups, val_groups, test_groups = splits['train'], splits['val'], splits['test']
    train_mask = np.isin(groups, train_groups)
    val_mask = np.isin(groups, val_groups)
    test_mask = np.isin(groups, test_groups)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  - Train Split: {len(y_train)} residues from {len(train_groups)} proteins")
    print(f"  - Val Split: {len(y_val)} residues from {len(val_groups)} proteins")
    print(f"  - Test Split: {len(y_test)} residues from {len(test_groups)} proteins")

    # 5. Normalize continuous features
    print("Normalizing continuous features...")
    feature_idxs = protein_data_list[0]['feature_idxs']
    
    # Define which feature groups are continuous and need scaling
    features_to_scale_names = ['embedding', 'b_factor', 'length', 'rsa', 'dist_to_glycosylation']
    
    cols_to_scale_indices = []
    for feature_name in features_to_scale_names:
        if feature_name in feature_idxs:
            cols_to_scale_indices.extend(list(feature_idxs[feature_name]))
            
    print(f"Identified {len(cols_to_scale_indices)} feature columns to scale.")

    scaler = StandardScaler()

    # Fit the scaler ONLY on the training data's continuous features
    scaler.fit(X_train[:, cols_to_scale_indices])

    # Transform the train, validation, and test sets using the fitted scaler
    X_train[:, cols_to_scale_indices] = scaler.transform(X_train[:, cols_to_scale_indices])
    X_val[:, cols_to_scale_indices] = scaler.transform(X_val[:, cols_to_scale_indices])
    X_test[:, cols_to_scale_indices] = scaler.transform(X_test[:, cols_to_scale_indices])
    
    print("Train, validation, and test sets successfully normalized.")

    # 6. Save the processed data into the datastore directory
    os.makedirs(config.DATASTORE_DIR, exist_ok=True)

    # Save the normalized feature matrices
    np.save(os.path.join(config.DATASTORE_DIR, 'X_num_train.npy'), X_train.astype(np.float32))
    np.save(os.path.join(config.DATASTORE_DIR, 'X_num_val.npy'), X_val.astype(np.float32))
    np.save(os.path.join(config.DATASTORE_DIR, 'X_num_test.npy'), X_test.astype(np.float32))

    # Save the labels
    np.save(os.path.join(config.DATASTORE_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.DATASTORE_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(config.DATASTORE_DIR, 'y_test.npy'), y_test)

    # Save the fitted scaler object for inference use
    scaler_path = os.path.join(config.DATASTORE_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler object saved to: {scaler_path}")

    # 7. Create and save the info.json metadata file
    info = {
        'name': 'epitope_prediction_streamlined_dataset',
        'task_type': 'binclass',
        'train_size': len(y_train),
        'val_size': len(y_val),
        'test_size': len(y_test),
        'n_num_features': X_train.shape[1],
        'n_cat_features': 0,
        'n_classes': 1,
        'normalization': 'StandardScaler',
        'normalization_details': {
            'columns_scaled_by_index': sorted(cols_to_scale_indices),
            'columns_scaled_by_name': features_to_scale_names
        }
    }

    info_path = os.path.join(config.DATASTORE_DIR, 'info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    print(f"Metadata saved to: {info_path}")
    print("--- Tabular Dataset Packaging Complete ---")

if __name__ == '__main__':
    normalize_and_save()
