# run_pipeline.py
import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from preprocessing import structure_cleaning
from preprocessing import sequence_clustering
from preprocessing import feature_extractor
from preprocessing import dataset_normalizer
from training import train_transformer


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End KO-BCE / gKO-BCE Conformational B-Cell Epitope Training Pipeline")
    parser.add_argument("--input_tsv", type=str, default=None,
                        help="Path to dataset TSV listing complexes and chains")
    parser.add_argument("--pdb_dir", type=str, default=None,
                        help="Directory containing the raw complex PDB structures")
    parser.add_argument("--use_ptms", action="store_true", default=False,
                        help="Incorporate the glycosylation-proximity feature (gKO-BCE; needs network access)")
    parser.add_argument("--skip_preprocessing", action="store_true", default=False,
                        help="Skip PDB cleaning, CD-HIT clustering and feature extraction")
    parser.add_argument("--skip_training", action="store_true", default=False,
                        help="Skip EpitopeTransformer training")
    args = parser.parse_args()

    tsv_path = Path(args.input_tsv) if args.input_tsv else config.INPUT_TSV
    pdb_dir_path = Path(args.pdb_dir) if args.pdb_dir else config.PDB_DIR

    print("\n" + "=" * 80)
    print("   KO-BCE / gKO-BCE END-TO-END REPRODUCIBILITY PIPELINE")
    print("=" * 80)
    print(f"Dataset TSV:   {tsv_path}")
    print(f"Raw PDBs:      {pdb_dir_path}")
    print(f"Model:         {'gKO-BCE (glyco feature ON)' if args.use_ptms else 'KO-BCE (glyco feature OFF)'}")
    print("=" * 80 + "\n")

    if not args.skip_preprocessing:
        if not tsv_path.exists():
            print(f"Error: input dataset TSV not found at {tsv_path}.")
            sys.exit(1)
        if not pdb_dir_path.exists():
            print(f"Error: raw PDB directory not found at {pdb_dir_path}.")
            sys.exit(1)

    # Create required working directories
    for path in [config.OUTPUT_DIR, config.CLEANED_PDB_DIR, config.ANTIGEN_ONLY_PDB_DIR,
                 config.EMBEDDING_CACHE_DIR, config.PCA_MODEL_CACHE_DIR, config.DATASTORE_DIR,
                 config.TRANSFORMER_MODEL_DIR]:
        os.makedirs(path, exist_ok=True)

    # ==========================================
    # STAGE 1: DATA PREPROCESSING
    # ==========================================
    if not args.skip_preprocessing:
        df = pd.read_csv(tsv_path, sep='\t') if tsv_path.suffix == '.tsv' else pd.read_csv(tsv_path)

        print("\n" + "#" * 40)
        print("  STAGE 1: STRUCTURE PROCESSING & CLUSTERING")
        print("#" * 40)

        # 1.1 Clean PDB structures (isolate complex vs antigen)
        structure_cleaning.clean_pdbs(
            df=df,
            pdb_dir=str(pdb_dir_path),
            cleaned_pdb_dir=str(config.CLEANED_PDB_DIR),
            antigen_only_pdb_dir=str(config.ANTIGEN_ONLY_PDB_DIR),
        )

        # 1.2 CD-HIT clustering + homology-aware splits
        print("\n--- Sequence Clustering and Cross-Validation Splitting ---")
        sequence_clustering.generate_splits(df, cdhit_threshold=config.CDHIT_THRESHOLD,
                                            max_size=config.MAX_CLUSTER_SIZE)

        # 1.3 Multi-modal feature extraction
        print("\n--- Multi-Modal Feature Extraction ---")
        feature_extractor.generate_features(
            df=df,
            cleaned_pdb_dir=str(config.CLEANED_PDB_DIR),
            antigen_only_pdb_dir=str(config.ANTIGEN_ONLY_PDB_DIR),
            is_training=True,
            use_ptms=args.use_ptms,
        )

        # 1.4 Normalize (QuantileTransformer) and package train/val/test matrices
        print("\n--- Tabular Dataset Normalization ---")
        dataset_normalizer.normalize_and_save()
    else:
        print("\n[INFO] Skipping Stage 1 Preprocessing.")

    # ==========================================
    # STAGE 2: MODEL TRAINING
    # ==========================================
    if not args.skip_training:
        print("\n" + "#" * 40)
        print("  STAGE 2: EPITOPETRANSFORMER TRAINING")
        print("#" * 40)
        train_transformer.run_transformer_training()
    else:
        print("\n[INFO] Skipping Stage 2 Training.")

    print("\n" + "=" * 80)
    print("   PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
