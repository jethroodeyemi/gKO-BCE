# xgboost_inference.py
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

class XGBoostInference:
    def __init__(self, model_path):
        """
        Initialize the XGBoost inference class.
        
        Args:
            model_path: Path to the trained .json model.
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
    def predict(self, X, threshold=0.6):
        """
        Runs prediction on continuous numerical feature matrix X.
        """
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        return predictions, probabilities

def run_inference(model_path, structured_data_path, threshold=0.6):
    """
    Runs XGBoost inference over the preprocessed structured protein features file.
    """
    print(f"Initializing XGBoost inference engine...")
    inference = XGBoostInference(model_path)
    
    print(f"Loading structured features from {structured_data_path}...")
    with open(structured_data_path, 'rb') as f:
        protein_data_list = pickle.load(f)
        
    results_list = []
    
    for protein_data in protein_data_list:
        pdb_id = protein_data['pdb_id']
        X = protein_data['X_arr']
        df_stats = protein_data['df_stats'].copy()
        
        predictions, probabilities = inference.predict(X, threshold=threshold)
        
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
