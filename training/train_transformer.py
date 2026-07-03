# train_transformer.py
import os
import json
import shutil
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt

import config
from models import EpitopeTransformer

# Fixed architecture for KO-BCE / gKO-BCE (matches the released checkpoints).
MODEL_CONFIG = {
    "token_bias": True,
    "n_layers": 3,
    "d_token": 256,
    "n_heads": 32,
    "attention_dropout": 0.3,
    "ffn_dropout": 0.1,
    "residual_dropout": 0.1,
    "prenormalization": True,
    "kv_compression": None,
    "kv_compression_sharing": None,
    "d_out": 1,
    "init_scale": 0.01,
    "activation": "tanglu",
}
WARMUP_EPOCHS = 10


def seed_everything(seed=config.SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description="Train EpitopeTransformer (KO-BCE / gKO-BCE)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.5, help="Beta parameter for hidden mixup")
    parser.add_argument("--no_mixup", action="store_true", help="Disable hidden mixup augmentation")
    # Parse known args so this is callable from run_pipeline without arg conflicts.
    args, _ = parser.parse_known_args()
    return args


def calculate_metrics(y_true, y_pred_proba):
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)
    return {
        "auc_pr": average_precision_score(y_true, y_pred_proba),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred_bin),
        "precision": precision_score(y_true, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true, y_pred_bin, zero_division=0),
    }


@torch.inference_mode()
def evaluate_dataloader(model, dataloader, device):
    model.eval()
    all_logits, all_labels, total_loss, total = [], [], 0.0, 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        total_loss += F.binary_cross_entropy_with_logits(logits, batch_y.float(), reduction="sum").item()
        total += batch_y.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(batch_y.cpu())
    labels = torch.cat(all_labels).numpy()
    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    metrics = calculate_metrics(labels, probs)
    metrics["loss"] = total_loss / total
    return metrics


def plot_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=200); plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["val_roc_auc"], label="Val AUC-ROC")
    plt.plot(epochs, history["val_auc_pr"], label="Val AUC-PR")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metric_curves.png"), dpi=200); plt.close()


def run_transformer_training():
    args = get_args()
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")
    use_mixup = not args.no_mixup

    datastore = config.DATASTORE_DIR
    X_train = np.load(os.path.join(datastore, "X_num_train.npy"))
    X_val = np.load(os.path.join(datastore, "X_num_val.npy"))
    X_test = np.load(os.path.join(datastore, "X_num_test.npy"))
    y_train = np.load(os.path.join(datastore, "y_train.npy")).astype(np.float32)
    y_val = np.load(os.path.join(datastore, "y_val.npy")).astype(np.float32)
    y_test = np.load(os.path.join(datastore, "y_test.npy")).astype(np.float32)

    with open(os.path.join(datastore, "info.json")) as f:
        n_features = json.load(f)["n_num_features"]
    print(f"Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}, Features={n_features}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=4096, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                             batch_size=4096, shuffle=False)

    model_cfg = {"d_numerical": n_features, **MODEL_CONFIG}
    model = EpitopeTransformer(**model_cfg).to(device)

    # No weight decay on tokenizer / norm / bias parameters.
    def needs_wd(name):
        return all(x not in name for x in ["tokenizer", ".norm", ".bias"])

    optimizer = torch.optim.AdamW([
        {"params": [v for k, v in model.named_parameters() if needs_wd(k)], "weight_decay": args.wd},
        {"params": [v for k, v in model.named_parameters() if not needs_wd(k)], "weight_decay": 0.0},
    ], lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - WARMUP_EPOCHS))

    out_dir = str(config.TRANSFORMER_MODEL_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print("\n--- Training EpitopeTransformer ---")
    best_val_roc_auc = -1.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_roc_auc": [], "val_auc_pr": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        # linear LR warmup, then cosine decay
        if epoch <= WARMUP_EPOCHS:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * epoch / WARMUP_EPOCHS
        else:
            scheduler.step()

        epoch_loss, n_batches = 0.0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            if use_mixup:
                logits, lam, shuffled_ids = model(batch_x, mixup=True, beta=args.beta)
                # lam is the per-sample Beta keep-rate; interpolate the two BCE targets.
                loss = (lam * F.binary_cross_entropy_with_logits(logits, batch_y, reduction="none")
                        + (1 - lam) * F.binary_cross_entropy_with_logits(logits, batch_y[shuffled_ids], reduction="none"))
                loss = loss.mean()
            else:
                loss = F.binary_cross_entropy_with_logits(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1

        val_metrics = evaluate_dataloader(model, val_loader, device)
        history["train_loss"].append(epoch_loss / n_batches)
        history["val_loss"].append(val_metrics["loss"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["val_auc_pr"].append(val_metrics["auc_pr"])

        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss {epoch_loss / n_batches:.4f} | "
              f"Val AUC-ROC {val_metrics['roc_auc']:.4f} | Val AUC-PR {val_metrics['auc_pr']:.4f}")

        # Early stopping / checkpointing on validation AUC-ROC (as in the paper).
        if val_metrics["roc_auc"] > best_val_roc_auc:
            best_val_roc_auc = val_metrics["roc_auc"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
            print(f"  --> best checkpoint saved (Val AUC-ROC {best_val_roc_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping (no Val AUC-ROC improvement for {args.patience} epochs).")
                break

    # Evaluate best checkpoint on the held-out test set.
    print("\n--- Evaluating best checkpoint on the hold-out test set ---")
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device))
    test_metrics = evaluate_dataloader(model, test_loader, device)
    print("=" * 46)
    print(f"  TEST AUC-ROC: {test_metrics['roc_auc']:.4f}")
    print(f"  TEST AUC-PR:  {test_metrics['auc_pr']:.4f}")
    print(f"  TEST F1:      {test_metrics['f1']:.4f}")
    print("=" * 46)

    # Write a self-contained checkpoint dir matching the released layout.
    results_json = {
        "config": {"normalization": config.NORMALIZATION, "beta": args.beta, "activation": "tanglu"},
        "cfg": {"model": {k: MODEL_CONFIG[k] for k in
                          ["prenormalization", "kv_compression", "kv_compression_sharing",
                           "token_bias", "activation"]}},
        "n_num_features": n_features,
        "n_classes": 1,
        "test_roc_auc": test_metrics["roc_auc"],
        "test_auc_pr": test_metrics["auc_pr"],
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=4)
    for fname in ("normalizer_quantile.pkl", "info.json"):
        src = os.path.join(config.DATASTORE_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fname))

    plot_curves(history, out_dir)


if __name__ == "__main__":
    run_transformer_training()
