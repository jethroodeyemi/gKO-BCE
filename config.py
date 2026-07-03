# config.py
import os
from pathlib import Path

# --- Central Base Directory ---
BASE_DIR = Path(__file__).resolve().parent

# --- File and Directory Paths ---
INPUT_TSV = BASE_DIR / 'data/dataset.tsv'
OUTPUT_DIR = BASE_DIR / 'data/processed'
PDB_DIR = BASE_DIR / 'data/pdb_files'
CLEANED_PDB_DIR = BASE_DIR / 'data/cleaned_pdb_files'
ANTIGEN_ONLY_PDB_DIR = BASE_DIR / 'data/antigen_only_pdb_files'
EMBEDDING_CACHE_DIR = BASE_DIR / 'data/embedding_cache'
PCA_MODEL_CACHE_DIR = BASE_DIR / 'models/pca_models'

# A small set of antigen-only structures shipped with the repo for a quick demo.
EXAMPLE_PDB_DIR = BASE_DIR / 'data/example_pdbs'

DEDUPED_TSV = OUTPUT_DIR / 'dataset_deduplicated.tsv'
FINAL_DATAFRAME_PATH = OUTPUT_DIR / 'antigen_residue_features.pkl'
STRUCTURED_DATA_PATH = OUTPUT_DIR / 'structured_protein_data.pkl'

# --- CD-HIT Sequence Clustering & Splits ---
FASTA_PATH = OUTPUT_DIR / 'all_antigen_sequences.fasta'
CLUSTER_FILE_PATH = OUTPUT_DIR / 'protein_clusters'  # CD-HIT will add .clstr
SPLITS_FILE_PATH = OUTPUT_DIR / 'split_clean.json'    # paper's homology-aware "clean" split
CDHIT_THRESHOLD = 0.4
MAX_CLUSTER_SIZE = 50

# --- Tabular Normalized Datastore ---
DATASTORE_DIR = BASE_DIR / 'data/datastore'

# --- Local Model Output Path (for re-training) ---
TRANSFORMER_MODEL_DIR = BASE_DIR / 'models/transformer'

# --- Feature Engineering Parameters ---
DISTANCE_THRESHOLD = 6.0
SASA_MAX_VALUES = {
    "A": 106.0, "R": 248.0, "N": 157.0, "D": 163.0, "C": 135.0, "Q": 198.0,
    "E": 194.0, "G": 84.0,  "H": 184.0, "I": 169.0, "L": 164.0, "K": 205.0,
    "M": 188.0, "F": 197.0, "P": 136.0, "S": 130.0, "T": 142.0, "W": 227.0,
    "Y": 222.0, "V": 142.0, "X": 169.55,
}

# --- Glycosylation Configuration (the "g" in gKO-BCE) ---
GLYCOSYLATION_MODE = ['distance']  # options: 'binary', 'distance'
MAX_GLYCOSYLATION_DISTANCE = 20.0  # tau: truncation distance for the glyco-proximity feature (Angstroms)

# --- Multi-Modal ESM Protein Language Models ---
EMBEDDING_MODE = ['esm2', 'esm_if1', 'esm1v']
FORCE_RECOMPUTE_EMBEDDINGS = False
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM_IF1_MODEL_NAME = "esm_if1_gvp4_t16_142M_UR50"
ESM1V_MODEL_NAME = "esm1v_t33_650M_UR90S_1"

# --- Dimensionality Reduction via PCA (1280 -> 256 for ESM-2 & ESM-1v) ---
REDUCE_ESM2_DIM = True
ESM2_DIM_TARGET = 256
REDUCE_ESM1V_DIM = True
ESM1V_DIM_TARGET = 256
REDUCE_ESM_IF1_DIM = False   # ESM-IF1 (512-d) is kept at full dimension
ESM_IF1_DIM_TARGET = 64

# --- Tabular Normalization (must match the released weights) ---
# The published KO-BCE / gKO-BCE checkpoints were trained with a QuantileTransformer
# (output_distribution='normal') fit on ALL feature columns. Inference reuses the fitted
# `normalizer_quantile.pkl` that ships with each checkpoint.
NORMALIZATION = 'quantile'
QUANTILE_NOISE = 1e-3
SEED = 42

# --- Inference Threshold ---
TRANSFORMER_THRESHOLD = 0.4

# ==========================================================================
#  Released model weights (hosted on the Hugging Face Hub)
# ==========================================================================
# The inference code downloads these on first use and caches them under
# WEIGHTS_CACHE_DIR. Point HF_REPO_ID / WEIGHTS_BASE_URL at any host that serves
# the same file layout (Hugging Face, Zenodo, a GitHub Release, ...).
HF_REPO_ID = "jethroodeyemi/gKO-BCE"
HF_REVISION = "main"
WEIGHTS_BASE_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/{HF_REVISION}"
WEIGHTS_CACHE_DIR = BASE_DIR / 'models/weights'

# Files that make up one downloadable checkpoint.
WEIGHT_FILES = ["best_model.pt", "normalizer_quantile.pkl", "results.json", "info.json"]

# The two published models share one transformer architecture; they differ only in
# whether the glycosylation-proximity feature is included (the "g" in gKO-BCE).
MODELS = {
    "ko-bce": {
        "remote_subdir": "ko-bce",
        "use_ptms": False,
        "n_num_features": 1047,
        "description": "KO-BCE: structure-aware BCE predictor (no glycosylation feature).",
    },
    "gko-bce": {
        "remote_subdir": "gko-bce",
        "use_ptms": True,
        "n_num_features": 1049,
        "description": "gKO-BCE: KO-BCE + glycosylation-proximity feature (flagship model).",
    },
}
DEFAULT_MODEL = "gko-bce"
