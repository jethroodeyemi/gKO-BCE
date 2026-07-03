# KO-BCE / gKO-BCE

**Structure-aware conformational B-cell epitope (BCE) prediction, robust to glycan shielding.**

This repository accompanies the paper *"gKO-BCE: A Novel Deep Learning Approach to
Predict B-Cell Epitopes, Even Within Glycosylated Antigens."* It provides the full,
reproducible pipeline — data preprocessing, feature extraction, model training, and
inference — for two models that share one transformer architecture:

| Model | Glycosylation feature | Independent test set (970 proteins) |
|-------|:---------------------:|:-----------------------------------:|
| **KO-BCE**  | ✗ | AUC-ROC 0.788 |
| **gKO-BCE** | ✓ (glycosylation-proximity) | **AUC-ROC 0.846, AUC-PR 0.217** |

The `g` in gKO-BCE denotes the geometric **glycosylation-proximity feature** that lets
the model down-weight glycan-shielded surface and recover true epitopes on heavily
glycosylated antigens (e.g. the SARS-CoV-2 spike).

> **Naming note.** In the code the transformer class is called `EpitopeTransformer`
> (in `models/transformer_arch.py`). It is the *same* network the paper refers to as
> **KO-BCE / gKO-BCE**; the two models differ only in the input feature set.

---

## Repository layout

```text
BCE_Pred_Streamlined/
├── README.md
├── LICENSE
├── requirements.txt
├── config.py                    # Central config: paths, feature params, weights registry (HF URLs)
├── run_pipeline.py              # End-to-end training/preprocessing orchestrator
│
├── data/
│   ├── dataset.tsv              # Deduplicated antibody–antigen index (pdb, Hchain, Lchain, antigen_chain, …)
│   ├── processed/               # Homology-aware split JSONs (split_clean.json = paper's main split)
│   └── example_pdbs/            # Small antigen-only structures for a quick inference demo
│
├── preprocessing/
│   ├── structure_cleaning.py    # Isolates H, L, and antigen chains
│   ├── sequence_clustering.py   # CD-HIT clustering → homology-aware splits
│   ├── glycosylation.py         # RCSB GraphQL + UniProt lookup of glycosylation sites
│   ├── esm_embedding.py         # ESM-2 / ESM-1v / ESM-IF1 embeddings
│   ├── feature_extractor.py     # Assembles per-residue feature matrices
│   └── dataset_normalizer.py    # QuantileTransformer normalization + train/val/test packaging
│
├── models/
│   ├── transformer_arch.py      # EpitopeTransformer (KO-BCE / gKO-BCE architecture)
│   └── pca_models/              # Fitted ESM-2/ESM-1v PCA reducers (1280→256), shipped
│
├── training/
│   └── train_transformer.py     # Trains with hidden mixup, warmup+cosine, early stopping on val AUC-ROC
│
└── inference/
    ├── main.py                  # CLI: predict epitopes on your PDBs (auto-downloads weights)
    ├── transformer_inference.py # Loads checkpoint + quantile normalizer, runs the model
    └── weights.py               # Fetches released weights from the Hugging Face Hub
```

---

## Installation

```bash
conda create -n gko-bce python=3.10 -y
conda activate gko-bce

# PyTorch (pick the build matching your CUDA runtime)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ESM-IF1 (inverse folding) needs PyTorch Geometric + scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-geometric

pip install -r requirements.txt

# Only needed to regenerate homology-aware splits during training:
conda install -c bioconda cd-hit -y
```

---

## Quick start — inference

Run the flagship **gKO-BCE** model on the bundled example antigens. The weights
(~20 MB) download automatically from the Hugging Face Hub on first use and are cached
under `models/weights/`:

```bash
python inference/main.py --model gko-bce
```

Predict on your own structures (antigen-only PDBs, or antibody–antigen complexes with
`--antigen_chain`):

```bash
# gKO-BCE (glycosylation-aware; looks up glyco sites for the PDB ID via RCSB/UniProt)
python inference/main.py --model gko-bce --input_pdb_dir path/to/pdbs --output_dir results/

# KO-BCE (no glycosylation feature, no network lookup)
python inference/main.py --model ko-bce  --input_pdb_dir path/to/pdbs --antigen_chain A
```

Key options: `--model {ko-bce,gko-bce}`, `--input_pdb_dir DIR`, `--antigen_chain ID`
(default: auto-detect the longest protein chain), `--threshold FLOAT` (default 0.4),
`--output_dir DIR`.

> **gKO-BCE requires network access** to look up glycosylation sites for each input
> (RCSB PDB GraphQL + UniProt REST), keyed by the PDB-ID filename. For structures with
> no PDB entry (e.g. AlphaFold models) the glyco feature defaults to "no nearby glycan";
> use `--model ko-bce` if you don't need it.

### Output

One CSV per structure (`<PDB_ID>_<model>_predictions.csv`), sorted by descending
probability:

| pdb_id | chain | res_id | residue | rsa | b_factor | length | probability | prediction |
|--------|-------|--------|---------|-----|----------|--------|-------------|------------|
| 1a14 | N | 368 | SER | 0.72 | 24.3 | 390 | 0.94 | 1 |
| 1a14 | N | 329 | ASP | 0.68 | 18.5 | 390 | 0.88 | 1 |

---

## Training from scratch

Training reproduces the paper's pipeline: ESM-2/ESM-1v/ESM-IF1 embeddings + biophysical
(RSA, B-factor) + optional glycosylation-proximity features → QuantileTransformer
normalization → transformer with hidden mixup, 10-epoch warmup + cosine schedule, and
early stopping on validation AUC-ROC.

```bash
# gKO-BCE (with the glycosylation feature); drop --use_ptms for KO-BCE
python run_pipeline.py --input_tsv data/dataset.tsv --pdb_dir data/pdb_files --use_ptms

# Re-run only the training stage on already-processed features:
python run_pipeline.py --skip_preprocessing --use_ptms
```

You supply `data/pdb_files/` (raw antibody–antigen complexes; re-downloadable from RCSB
by PDB ID). The pipeline cleans structures, builds CD-HIT (40% identity) homology-aware
splits, extracts features, and trains. Fitted PCA reducers are shipped in
`models/pca_models/`; regenerate them only if you change the embedding set.

---

## Model weights

The released **KO-BCE** and **gKO-BCE** checkpoints are hosted on the Hugging Face Hub
and pulled automatically by the inference code:

- **Hub:** [`jethroodeyemi/gKO-BCE`](https://huggingface.co/jethroodeyemi/gKO-BCE)

Each checkpoint bundles `best_model.pt`, the fitted `normalizer_quantile.pkl`,
`results.json`, and `info.json`. The hosting is URL-based and host-agnostic: point
`HF_REPO_ID` / `WEIGHTS_BASE_URL` in `config.py` at any static host (Hugging Face,
Zenodo, a GitHub Release) that serves the same `ko-bce/…` and `gko-bce/…` layout.

---

## Data availability

Structures and annotations are derived from the **Structural Antibody Database
(SAbDab)** and the **RCSB Protein Data Bank**. Epitope residues are antigen residues
within **6 Å** of the antibody. Homology-aware train/val/test splits were built by
**CD-HIT at 40% sequence identity** (cluster-level split; `data/processed/split_clean.json`
is the paper's main split: 3,047 / 350 / 970 proteins). `data/dataset.tsv` lists the
antibody–antigen complexes and chains. Raw PDB files are not redistributed here; they
are re-downloadable from RCSB by the PDB IDs in `dataset.tsv`.

---

## Citation

```bibtex
@article{odeyemi_gkobce,
  title   = {gKO-BCE: A Novel Deep Learning Approach to Predict B-Cell Epitopes,
             Even Within Glycosylated Antigens},
  author  = {Odeyemi, Jethro and Kashyap, Monika and Wilson, Heather L. and Khatooni, Zahed},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
