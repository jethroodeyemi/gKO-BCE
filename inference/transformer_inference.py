# transformer_inference.py
import pickle
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

import config
from models import EpitopeTransformer

class TransformerInference:
    def __init__(self, model_path, results_cfg_path, scaler_path=None):
        """
        Initialize the Transformer inference class.
        
        Args:
            model_path: Path to the trained best_model.pt weights.
            results_cfg_path: Path to the results.json metadata from training.
            scaler_path: Path to the scaler.pkl file from training.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load training metadata and model configuration
        with open(results_cfg_path, 'r') as f:
            self.results = json.load(f)
            
        train_args = self.results.get('config', {})
        model_cfg = self.results.get('cfg', {}).get('model', {})
        n_features = self.results.get('n_num_features')
        
        # Load fitted StandardScaler
        self.scaler = None
        self.cols_to_scale_indices = None
        if scaler_path and Path(scaler_path).exists():
            print(f"Loading fitted scaler from {scaler_path}...")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            # Fetch scale indices from metadata
            with open(Path(scaler_path).parent / 'info.json', 'r') as f:
                info = json.load(f)
                self.cols_to_scale_indices = info['normalization_details']['columns_scaled_by_index']
                
        # Initialize model architecture
        self.model = EpitopeTransformer(**model_cfg).to(self.device)
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        # Handle DataParallel wrapping if present
        if state_dict and 'module.' in list(state_dict.keys())[0]:
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def predict(self, X, threshold=0.4, batch_size=4096):
        """
        Runs prediction on continuous numerical feature matrix X.
        """
        X_eval = X.copy()
        
        # Apply normalization using the loaded scaler
        if self.scaler is not None and self.cols_to_scale_indices is not None:
            X_eval[:, self.cols_to_scale_indices] = self.scaler.transform(X_eval[:, self.cols_to_scale_indices])
            
        X_eval = X_eval.astype(np.float32)
        
        # Build dataset and loader
        X_tensor = torch.from_numpy(X_eval).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        logits_list = []
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0]
                logits = self.model(x_batch)
                logits_list.append(logits.cpu())
                
        logits = torch.cat(logits_list)
        probabilities = torch.sigmoid(logits).numpy()
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities

def run_inference(model_path, results_cfg_path, scaler_path, structured_data_path, threshold=0.4, batch_size=4096):
    """
    Runs EpitopeTransformer inference over the preprocessed structured protein features file.
    """
    print(f"Initializing EpitopeTransformer inference engine...")
    inference = TransformerInference(model_path, results_cfg_path, scaler_path)
    
    print(f"Loading structured features from {structured_data_path}...")
    with open(structured_data_path, 'rb') as f:
        protein_data_list = pickle.load(f)
        
    results_list = []
    
    for protein_data in protein_data_list:
        pdb_id = protein_data['pdb_id']
        X = protein_data['X_arr']
        df_stats = protein_data['df_stats'].copy()
        
        predictions, probabilities = inference.predict(X, threshold=threshold, batch_size=batch_size)
        
        df_stats['prediction'] = predictions
        df_stats['probability'] = probabilities
        
        results_list.append(df_stats)
        
        print(f"\nPredictions for {pdb_id} (Top 10 residues with highest probability):")
        sorted_df = df_stats.sort_values(by='probability', ascending=False)
        for idx, row in sorted_df.head(10).iterrows():
            res_id = row['res_id']
            residue = row['residue']
            prob = row['probability']
            pred = row['prediction']
            print(f"  Residue {res_id} ({residue}): Prob={prob:.4f} | Prediction={pred}")
            
    return results_list
