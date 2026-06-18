# main.py
import os
import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from preprocessing import structure_cleaning
from preprocessing import feature_extractor
from inference import xgboost_inference
from inference import transformer_inference

def main():
    parser = argparse.ArgumentParser(description="Unified B-Cell Epitope Prediction Pipeline")
    parser.add_argument("--model", type=str, default="transformer", choices=["xgboost", "transformer"],
                        help="The trained model type to use for predictions")
    parser.add_argument("--input_pdb_dir", type=str, default=None,
                        help="Directory containing raw PDB files for inference")
    parser.add_argument("--antigen_chain", type=str, default=None,
                        help="Chain ID of the antigen in the input PDB files (if a single chain)")
    parser.add_argument("--use_ptms", action="store_true", default=False,
                        help="Include PTM (glycosylation) features (requires network connection to RCSB/UniProt)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Classification probability threshold (default: model-specific)")
    parser.add_argument("--output_dir", type=str, default="predictions",
                        help="Directory where per-residue prediction CSVs will be saved")
    args = parser.parse_args()

    # Determine paths and thresholds based on model selection
    if args.model == "xgboost":
        model_path = os.path.join(config.XGBOOST_MODEL_DIR, "final_model.json")
        threshold = args.threshold if args.threshold is not None else config.XGBOOST_THRESHOLD
    else:
        model_path = os.path.join(config.TRANSFORMER_MODEL_DIR, "best_model.pt")
        threshold = args.threshold if args.threshold is not None else config.TRANSFORMER_THRESHOLD

    results_cfg_path = os.path.join(config.TRANSFORMER_MODEL_DIR, "results.json")
    scaler_path = os.path.join(config.DATASTORE_DIR, "scaler.pkl")

    # 1. Prepare inputs
    input_dir = Path(args.input_pdb_dir) if args.input_pdb_dir else Path(config.PDB_DIR)
    if not input_dir.exists() or not list(input_dir.glob("*.pdb")):
        print(f"Error: Input PDB directory '{input_dir}' does not exist or contains no PDB files.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"Running BCE Prediction Inference Pipeline")
    print(f"Model Type:   {args.model.upper()}")
    print(f"Model Weights: {model_path}")
    print(f"Input PDBs:   {input_dir}")
    print(f"PTM Features: {'Enabled' if args.use_ptms else 'Disabled'}")
    print(f"Threshold:    {threshold}")
    print(f"=======================================================\n")

    # Clean PDB structures to extract antigen only and build structured list
    pdb_files = list(input_dir.glob("*.pdb"))
    df_list = []
    
    # We will assume a simple single chain or build rows to isolate antigen
    for pdb_file in pdb_files:
        pdb_id = pdb_file.stem
        # If antigen_chain is specified, use it. Otherwise, look for chain 'A' as standard default
        antigen_chain = args.antigen_chain if args.antigen_chain else "A"
        df_list.append({
            'pdb': pdb_id,
            'Hchain': 'H',
            'Lchain': 'L',
            'antigen_chain': antigen_chain
        })
    df_pdbs = pd.DataFrame(df_list)

    # 2. Extract Features
    # Create temp directory for sanitized structures
    temp_antigen_only_dir = config.OUTPUT_DIR / "temp_antigen_only"
    os.makedirs(temp_antigen_only_dir, exist_ok=True)
    
    print("Sanitizing structural files...")
    structure_cleaning.clean_pdbs(df_pdbs, str(input_dir), str(config.OUTPUT_DIR / "temp_cleaned"), str(temp_antigen_only_dir))

    print("Extracting per-residue features...")
    # Extract features (not training, so no label mapping)
    feature_extractor.generate_features(
        df_pdbs, 
        cleaned_pdb_dir=str(config.OUTPUT_DIR / "temp_cleaned"),
        antigen_only_pdb_dir=str(temp_antigen_only_dir),
        is_training=False,
        use_ptms=args.use_ptms
    )

    # 3. Run Model Inference
    if args.model == "xgboost":
        if not os.path.exists(model_path):
            print(f"Error: Trained model weights not found at {model_path}. Run training first.")
            sys.exit(1)
            
        all_results = xgboost_inference.run_inference(
            model_path=model_path,
            structured_data_path=str(config.STRUCTURED_DATA_PATH),
            threshold=threshold
        )
    else:
        if not os.path.exists(model_path) or not os.path.exists(results_cfg_path):
            print(f"Error: Trained weights or config metadata not found. Run training first.")
            sys.exit(1)
            
        all_results = transformer_inference.run_inference(
            model_path=model_path,
            results_cfg_path=results_cfg_path,
            scaler_path=scaler_path if os.path.exists(scaler_path) else None,
            structured_data_path=str(config.STRUCTURED_DATA_PATH),
            threshold=threshold
        )

    # 4. Save Predictions
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving per-residue prediction CSVs...")
    for df in all_results:
        pdb_id = df['pdb_id'].iloc[0]
        csv_output_path = out_dir / f"{pdb_id}_bce_predictions.csv"
        
        # Sort by probability descending to highlight highest-scoring epitopes
        sorted_df = df.sort_values(by='probability', ascending=False)
        sorted_df.to_csv(csv_output_path, index=False)
        
        positive_residues = (sorted_df['prediction'] == 1).sum()
        total_residues = len(sorted_df)
        print(f"  Saved predictions for {pdb_id} to: {csv_output_path} ({positive_residues}/{total_residues} residues predicted as epitopes)")

    print(f"\nInference Pipeline Complete. All results saved under '{out_dir}/'.")

if __name__ == '__main__':
    main()
