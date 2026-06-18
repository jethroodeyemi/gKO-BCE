# train_transformer.py
import os
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

import config
from models import EpitopeTransformer

def seed_everything(seed=42):
    """Sets standard random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description="Train EpitopeTransformer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta distribution parameter for Mixup")
    parser.add_argument("--mixup", action="store_true", default=True, help="Enable hidden mixup data augmentation")
    return parser.parse_args()

def calculate_metrics(y_true, y_pred_proba):
    """Computes binary classification evaluation metrics."""
    auc_pr = average_precision_score(y_true, y_pred_proba)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    # Threshold predictions at 0.5 for standard metrics
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    
    return {
        'auc_pr': auc_pr,
        'roc_auc': auc_roc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

@torch.inference_mode()
def evaluate_dataloader(model, dataloader, device):
    """Evaluates the model on a dataloader, returning the loss and predictions."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        loss = F.binary_cross_entropy_with_logits(logits, batch_y.float(), reduction='sum')
        total_loss += loss.item()
        total_samples += batch_y.size(0)
        
        all_logits.append(logits.cpu())
        all_labels.append(batch_y.cpu())
        
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probabilities = torch.sigmoid(logits).numpy()
    
    mean_loss = total_loss / total_samples
    metrics = calculate_metrics(labels, probabilities)
    metrics['loss'] = mean_loss
    
    return metrics, probabilities

def plot_curves(train_history, output_dir):
    """Plots and saves loss and metric history."""
    epochs = range(1, len(train_history['train_loss']) + 1)
    
    # 1. Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_history['train_loss'], label='Train Loss')
    plt.plot(epochs, train_history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300)
    plt.close()
    
    # 2. AUC-PR and AUC-ROC Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_history['train_auc_pr'], label='Train AUC-PR')
    plt.plot(epochs, train_history['val_auc_pr'], label='Val AUC-PR')
    plt.plot(epochs, train_history['val_auc_roc'], label='Val AUC-ROC', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_curves.png'), dpi=300)
    plt.close()

def run_transformer_training():
    args = get_args()
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using training device: {device}")
    
    # 1. Load Prepackaged Datasets
    datastore = config.DATASTORE_DIR
    print(f"Loading normalized dataset from {datastore}")
    
    X_train = np.load(os.path.join(datastore, 'X_num_train.npy'))
    X_val = np.load(os.path.join(datastore, 'X_num_val.npy'))
    X_test = np.load(os.path.join(datastore, 'X_num_test.npy'))
    
    y_train = np.load(os.path.join(datastore, 'y_train.npy'))
    y_val = np.load(os.path.join(datastore, 'y_val.npy'))
    y_test = np.load(os.path.join(datastore, 'y_test.npy'))
    
    with open(os.path.join(datastore, 'info.json'), 'r') as f:
        metadata = json.load(f)
        
    n_features = metadata['n_num_features']
    print(f"Dataset summary: Train size={len(y_train)}, Val size={len(y_val)}, Test size={len(y_test)}, Features={n_features}")

    # 2. Build PyTorch Dataloaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    # 3. Initialize Model and Optimizer
    model_cfg = {
        'd_numerical': n_features,
        'token_bias': True,
        'n_layers': 3,
        'd_token': 256,
        'n_heads': 32,
        'attention_dropout': 0.3,
        'ffn_dropout': 0.1,
        'residual_dropout': 0.1,
        'prenormalization': True,
        'd_out': 1,
        'init_scale': 0.01,
        'activation': 'tanglu'
    }
    
    model = EpitopeTransformer(**model_cfg).to(device)
    
    # Separate weight decay for parameters
    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])
    
    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    
    optimizer = torch.optim.AdamW([
        {'params': parameters_with_wd, 'weight_decay': args.wd},
        {'params': parameters_without_wd, 'weight_decay': 0.0}
    ], lr=args.lr)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    os.makedirs(config.TRANSFORMER_MODEL_DIR, exist_ok=True)

    # Save training configuration
    results_json = {
        'config': {
            'normalization': 'standard',
            'beta': args.beta,
            'activation': 'tanglu'
        },
        'cfg': {
            'model': model_cfg
        },
        'n_num_features': n_features,
        'n_classes': 1
    }
    with open(os.path.join(config.TRANSFORMER_MODEL_DIR, "results.json"), 'w') as f:
        json.dump(results_json, f, indent=4)

    # 4. Training Loop with Early Stopping
    print("\n--- Training EpitopeTransformer Model ---")
    best_val_auc_pr = -1.0
    epochs_no_improve = 0
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_auc_pr': [], 'val_auc_pr': [], 'val_auc_roc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            if args.mixup:
                # Run with hidden dimension mixup augmentation
                logits, feat_masks, shuffled_ids = model(batch_x, mixup=True, beta=args.beta)
                
                # Formulate soft label for mixup backpropagation
                # If a dimension mask maps back to its source vs shuffled index, we interpolate
                lam = feat_masks.mean(dim=-1) # average keep rate per token
                y_shuffled = batch_y[shuffled_ids]
                soft_labels = lam * batch_y + (1 - lam) * y_shuffled
                
                loss = F.binary_cross_entropy_with_logits(logits, soft_labels.float())
            else:
                logits = model(batch_x)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y.float())
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
        scheduler.step()
        
        # Epoch metrics
        mean_epoch_loss = epoch_loss / n_batches
        train_eval_metrics, _ = evaluate_dataloader(model, train_loader, device)
        val_metrics, _ = evaluate_dataloader(model, val_loader, device)
        
        history['train_loss'].append(mean_epoch_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_auc_pr'].append(train_eval_metrics['auc_pr'])
        history['val_auc_pr'].append(val_metrics['auc_pr'])
        history['val_auc_roc'].append(val_metrics['roc_auc'])
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {mean_epoch_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val AUC-PR: {val_metrics['auc_pr']:.4f} | Val AUC-ROC: {val_metrics['roc_auc']:.4f}")
        
        # Check validation improvement for checkpoint saving
        if val_metrics['auc_pr'] > best_val_auc_pr:
            best_val_auc_pr = val_metrics['auc_pr']
            epochs_no_improve = 0
            
            # Save best checkpoint
            torch.save(model.state_dict(), os.path.join(config.TRANSFORMER_MODEL_DIR, "best_model.pt"))
            print(f"  --> Best model checkpoint saved (Val AUC-PR: {best_val_auc_pr:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered. No validation improvement for {args.patience} epochs.")
                break
                
    # 5. Evaluate Best Checkpoint on Test Set
    print("\n--- Training Finished. Evaluating Best Checkpoint on Hold-Out Test Set ---")
    best_weights = torch.load(os.path.join(config.TRANSFORMER_MODEL_DIR, "best_model.pt"), map_location=device)
    model.load_state_dict(best_weights)
    
    test_metrics, _ = evaluate_dataloader(model, test_loader, device)
    print("==============================================")
    print(f"  TRANSFORMER TEST SET AUC-PR:  {test_metrics['auc_pr']:.4f}")
    print(f"  TRANSFORMER TEST SET AUC-ROC: {test_metrics['roc_auc']:.4f}")
    print(f"  TRANSFORMER TEST SET F1:      {test_metrics['f1']:.4f}")
    print("==============================================")
    
    # Save curves
    plot_curves(history, config.TRANSFORMER_MODEL_DIR)

if __name__ == '__main__':
    run_transformer_training()
