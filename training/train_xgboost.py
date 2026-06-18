# train_xgboost.py
import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

import config

def train_cv(X_train_val, y_train_val, groups_train_val):
    """
    Performs GroupKFold cross-validation on the training+validation set.
    """
    print("\n--- Starting XGBoost GroupKFold Cross-Validation ---")
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 5000,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'tree_method': 'hist',
        'random_state': 42
    }
    
    if torch_cuda_available := xgb.compat.SUPPORT_CUDA:
        print("Using GPU acceleration (CUDA) for XGBoost.")
        params['device'] = 'cuda'

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    
    models = []
    auc_pr_scores = []
    auc_roc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_val, y_train_val, groups=groups_train_val)):
        print(f"Fold {fold+1}/{n_splits}")
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        # Address class imbalance
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        
        model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        auc_pr = average_precision_score(y_val, y_pred_proba)
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        
        auc_pr_scores.append(auc_pr)
        auc_roc_scores.append(auc_roc)
        models.append(model)
        
        print(f"  Fold {fold+1} AUC-PR: {auc_pr:.4f} | AUC-ROC: {auc_roc:.4f}")

    print("\n--- Cross-Validation Results ---")
    print(f"Mean AUC-PR: {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")
    print(f"Mean AUC-ROC: {np.mean(auc_roc_scores):.4f} ± {np.std(auc_roc_scores):.4f}")
    
    best_fold_idx = np.argmax(auc_pr_scores)
    print(f"Best Fold: Fold {best_fold_idx + 1} with AUC-PR: {auc_pr_scores[best_fold_idx]:.4f}")

    return models, best_fold_idx

def train_final_model(X_train_val, y_train_val, best_params, best_iteration):
    """Trains a final model on all training + validation data."""
    print("\n--- Training Final XGBoost Model on Full Train/Val Dataset ---")
    
    final_params = best_params.copy()
    final_params.pop('scale_pos_weight', None)
    final_params['n_estimators'] = best_iteration
    final_params.pop('early_stopping_rounds', None)
    
    scale_pos_weight = np.sum(y_train_val == 0) / np.sum(y_train_val == 1)
    
    final_model = xgb.XGBClassifier(**final_params, scale_pos_weight=scale_pos_weight)
    final_model.fit(X_train_val, y_train_val, verbose=False)
    
    print("Final model training complete.")
    return final_model

def save_feature_importance_plot(model, feature_names, plot_path, top_n=30):
    """Generates and saves a feature importance plot."""
    print(f"Saving feature importance plot to {plot_path}...")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

def run_xgboost_pipeline():
    # 1. Load structured pickle data
    print("Loading structured protein features...")
    with open(config.STRUCTURED_DATA_PATH, 'rb') as f:
        protein_data_list = pickle.load(f)

    # 2. Reconstruct arrays
    features, labels, groups = [], [], []
    for p_data in tqdm(protein_data_list, desc="Reconstructing flat arrays"):
        features.append(p_data['X_arr'])
        labels.append(p_data['df_stats']['is_epitope'].values)
        groups.append(np.full(p_data['length'], p_data['pdb_id']))

    X = np.vstack(features)
    y = np.concatenate(labels)
    groups = np.concatenate(groups)

    # Resolve feature names
    feature_names = []
    feature_idxs = protein_data_list[0]['feature_idxs']
    sorted_features = sorted(feature_idxs.items(), key=lambda x: x[1].start)
    for name, idx_range in sorted_features:
        feature_names.extend([f"{name}_{i}" for i in range(len(idx_range))] if len(idx_range) > 1 else [name])

    # 3. Load splits
    print(f"Loading sequence cluster-based splits from {config.SPLITS_FILE_PATH}")
    with open(config.SPLITS_FILE_PATH, 'r') as f:
        splits = json.load(f)
        
    train_val_groups = splits['train'] + splits['val']
    test_groups = splits['test']

    train_val_mask = np.isin(groups, train_val_groups)
    test_mask = np.isin(groups, test_groups)

    X_train_val, X_test = X[train_val_mask], X[test_mask]
    y_train_val, y_test = y[train_val_mask], y[test_mask]
    groups_train_val = groups[train_val_mask]

    print(f"Train/Val set size: {len(X_train_val)} residues from {len(np.unique(groups_train_val))} proteins")
    print(f"Test set size: {len(X_test)} residues from {len(np.unique(groups[test_mask]))} proteins")

    # 4. Train with Cross-Validation
    cv_models, best_idx = train_cv(X_train_val, y_train_val, groups_train_val)
    best_cv_model = cv_models[best_idx]
    
    os.makedirs(config.XGBOOST_MODEL_DIR, exist_ok=True)
    best_cv_model.save_model(os.path.join(config.XGBOOST_MODEL_DIR, "best_cv_model.json"))
    print(f"Saved best CV fold model to {os.path.join(config.XGBOOST_MODEL_DIR, 'best_cv_model.json')}")

    # 5. Train Final Unified Model
    best_params = best_cv_model.get_params()
    best_iteration = best_cv_model.best_iteration if hasattr(best_cv_model, 'best_iteration') else 100
    final_model = train_final_model(X_train_val, y_train_val, best_params, best_iteration)
    
    final_model_path = os.path.join(config.XGBOOST_MODEL_DIR, "final_model.json")
    final_model.save_model(final_model_path)
    print(f"Saved final unified model to {final_model_path}")

    # 6. Evaluate on hold-out Test Set
    print("\n--- Evaluation on Homology-Isolated Test Set ---")
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]

    test_auc_pr = average_precision_score(y_test, y_test_pred_proba)
    test_auc_roc = roc_auc_score(y_test, y_test_pred_proba)

    print("==============================================")
    print(f"  FINAL TEST SET AUC-PR:  {test_auc_pr:.4f}")
    print(f"  FINAL TEST SET AUC-ROC: {test_auc_roc:.4f}")
    print("==============================================")

    # 7. Feature Importance Plot
    save_feature_importance_plot(
        final_model, 
        feature_names, 
        os.path.join(config.XGBOOST_MODEL_DIR, "feature_importance.png")
    )

if __name__ == '__main__':
    run_xgboost_pipeline()
