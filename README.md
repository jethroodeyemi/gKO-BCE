#  Conformational B-Cell Epitope Prediction Pipeline
[![DOI](https://zenodo.org/badge/1273753930.svg)](https://doi.org/10.5281/zenodo.20753198)



```text
BCE_Pred_Streamlined/
│
├── README.md                     
├── requirements.txt              # Python dependencies
├── config.py                     # Centralized pipeline configurations and paths
├── run_pipeline.py               # Training and split orchestrator
│
├── preprocessing/                # Data cleaning, feature extraction, and splitting
│   ├── __init__.py
│   ├── structure_cleaning.py     # Isolates H, L, and Antigen chains
│   ├── sequence_clustering.py    # CD-HIT FASTA parser & homology split generator
│   ├── glycosylation.py          # Queries RCSB/UniProt for glycosylation sites
│   ├── esm_embedding.py          # Inferences ESM-2, ESM-1v, and ESM-IF1 models
│   ├── feature_extractor.py      
│   └── dataset_normalizer.py    
│
├── models/                       # Deep learning network architectures
│   ├── __init__.py
│   └── transformer_arch.py      
│
├── training/                     # Model fitting and cross-validation scripts
│   ├── __init__.py
│   ├── train_xgboost.py          # Trains XGBoost with CV & feature importance
│   └── train_transformer.py      # Optimizes EpitopeTransformer with Mixup
│
└── inference/                    # Model evaluation & deployment utilities
    ├── __init__.py
    ├── xgboost_inference.py      
    ├── transformer_inference.py  
    └── main.py                   
```

---

## Requirements and Installation

### 1. Python Environment

We recommend Python 3.8 to 3.11 with a virtual environment manager:

```bash
conda create -n bce-pred python=3.9 -y
conda activate bce-pred
```

### 2. PyTorch & CUDA Support

For CUDA 12.1 runtime, execute:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Core Dependencies


```bash
pip install -r requirements.txt
```

### 4. Sequence Clustering Tools (CD-HIT)

CD-HIT is required to compute homology-aware splits.

```bash
conda install -c bioconda cd-hit -y
```

---

## Running the Pipeline

### 1. Preparation

To begin, ensure your input dataset file is specified at `data/dataset.tsv` and raw PDB complex files are present in `data/pdb_files/`.
The input TSV should contain the following tab-separated columns:
- `pdb`: The 4-character PDB complex ID (e.g., `1a14`).
- `Hchain`: Heavy antibody chain ID.
- `Lchain`: Light antibody chain ID.
- `antigen_chain`: Target antigen chain ID.

### 2. Execution

Run the training and split pipeline by executing:

```bash
python run_pipeline.py --input_tsv data/dataset.tsv --pdb_dir data/pdb_files
```

To incorporate glycosylation mapping features:

```bash
python run_pipeline.py --use_ptms
```

To skip the feature extraction stages and re-run only the ML training:

```bash
python run_pipeline.py --skip_preprocessing
```

---

## Running Inference on Custom Proteins

### 1. Run with EpitopeTransformer

```bash
python inference/main.py --model transformer --input_pdb_dir path/to/pdb_folder --output_dir results_dir
```

### 2. Run with XGBoost

```bash
python inference/main.py --model xgboost --input_pdb_dir path/to/pdb_folder --output_dir results_dir
```

### Output Format

For each PDB structure evaluated, a CSV file named `<PDB_ID>_bce_predictions.csv` will be generated. The spreadsheet contains per-residue prediction probabilities and is sorted descending by score:

| pdb_id | chain | res_id | residue | rsa | b_factor | probability | prediction |
|---|---|---|---|---|---|---|---|
| 1a14 | A | 142 | ASN | 0.8124 | 24.32 | 0.9412 | 1 |
| 1a14 | A | 88  | LYS | 0.7410 | 18.55 | 0.8950 | 1 |
| 1a14 | A | 12  | VAL | 0.1235 | 12.10 | 0.0420 | 0 |
