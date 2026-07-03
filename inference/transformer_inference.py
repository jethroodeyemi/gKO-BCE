# transformer_inference.py
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import EpitopeTransformer

# Model hyper-parameters that are NOT stored in results.json but are fixed for every
# released KO-BCE / gKO-BCE checkpoint (they match the training defaults).
DEFAULT_MODEL_CONFIG = {
    "ffn_dropout": 0.1,
    "attention_dropout": 0.3,
    "residual_dropout": 0.1,
    "n_layers": 3,
    "n_heads": 32,
    "d_token": 256,
    "init_scale": 0.01,
}


class TransformerInference:
    """Loads a released checkpoint (weights + fitted quantile normalizer) and predicts."""

    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = Path(model_dir)

        # --- Training metadata / model configuration ---
        with open(model_dir / "results.json", "r") as f:
            self.results = json.load(f)
        with open(model_dir / "info.json", "r") as f:
            info = json.load(f)

        n_num_features = info.get("n_num_features")
        d_out = info.get("n_classes") or 1
        cfg_model = self.results.get("cfg", {}).get("model", {})

        kwargs = {"d_numerical": n_num_features, "d_out": d_out, **cfg_model}
        for k, v in DEFAULT_MODEL_CONFIG.items():
            kwargs.setdefault(k, v)

        # --- Build architecture and load weights ---
        self.model = EpitopeTransformer(**kwargs).to(self.device)
        state_dict = torch.load(model_dir / "best_model.pt", map_location=self.device)
        if state_dict and list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # --- Fitted QuantileTransformer (applied to ALL feature columns) ---
        normalizer_path = model_dir / "normalizer_quantile.pkl"
        self.normalizer = None
        if normalizer_path.exists():
            with open(normalizer_path, "rb") as f:
                self.normalizer = pickle.load(f)
        else:
            print(f"WARNING: {normalizer_path} not found; predictions will be unreliable.")

    def predict(self, X, threshold=0.4, batch_size=8):
        X = np.asarray(X, dtype=np.float32)
        if self.normalizer is not None:
            X = self.normalizer.transform(X).astype(np.float32)

        loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
        logits = []
        with torch.no_grad():
            for (batch,) in loader:
                logits.append(self.model(batch.to(self.device)).cpu())
        probabilities = torch.sigmoid(torch.cat(logits)).numpy()
        predictions = (probabilities >= threshold).astype(int)
        return predictions, probabilities


def run_inference(model_dir, structured_data_path, threshold=0.4, batch_size=8, top_k=10):
    """Run EpitopeTransformer inference over the preprocessed structured protein features."""
    print("Initializing EpitopeTransformer inference engine...")
    engine = TransformerInference(model_dir)

    print(f"Loading structured features from {structured_data_path}...")
    with open(structured_data_path, "rb") as f:
        protein_data_list = pickle.load(f)

    results_list = []
    for protein_data in protein_data_list:
        pdb_id = protein_data["pdb_id"]
        X = protein_data["X_arr"]
        df_stats = protein_data["df_stats"].copy()

        predictions, probabilities = engine.predict(X, threshold=threshold, batch_size=batch_size)
        df_stats["prediction"] = predictions
        df_stats["probability"] = probabilities
        results_list.append(df_stats)

        sorted_df = df_stats.sort_values(by="probability", ascending=False)
        n_pos = int((df_stats["prediction"] == 1).sum())
        print(f"\n{pdb_id}: {n_pos}/{len(df_stats)} residues predicted as epitope "
              f"(threshold={threshold}). Top {top_k}:")
        for _, row in sorted_df.head(top_k).iterrows():
            print(f"  Residue {row['res_id']} ({row['residue']}): "
                  f"prob={row['probability']:.4f} pred={row['prediction']}")

    return results_list
