# main.py
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select, Polypeptide

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from preprocessing import feature_extractor
from inference import transformer_inference
from inference import weights


class _SingleChainSelect(Select):
    """Keep only one chain and drop heteroatoms/waters/glycans."""
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id

    def accept_residue(self, residue):
        return 1 if residue.get_id()[0] == ' ' else 0


def detect_antigen_chain(structure):
    """Pick the longest standard-amino-acid chain as the antigen."""
    best_id, best_len = None, -1
    for chain in structure[0]:
        n = sum(1 for res in chain if Polypeptide.is_aa(res, standard=True))
        if n > best_len:
            best_id, best_len = chain.get_id(), n
    return best_id if best_len > 0 else None


def prepare_antigen_only(input_dir, out_dir, antigen_chain=None):
    """Extract a clean antigen-only structure per input PDB and return the index DataFrame.

    RSA/SASA and embeddings must be computed on the isolated antigen (never in the
    antibody complex), so we always re-extract the chosen chain here.
    """
    parser = PDBParser(QUIET=True)
    io = PDBIO()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pdb_file in sorted(input_dir.glob("*.pdb")):
        pdb_id = pdb_file.stem
        try:
            structure = parser.get_structure(pdb_id, str(pdb_file))
        except Exception as e:
            print(f"  Skipping {pdb_id}: could not parse ({e}).")
            continue

        chain_id = antigen_chain if antigen_chain else detect_antigen_chain(structure)
        if chain_id is None or chain_id not in structure[0]:
            print(f"  Skipping {pdb_id}: no usable antigen chain (requested '{antigen_chain}').")
            continue

        out_path = out_dir / f"{pdb_id}_antigen_only.pdb"
        io.set_structure(structure)
        io.save(str(out_path), _SingleChainSelect(chain_id))
        rows.append({"pdb": pdb_id, "Hchain": "", "Lchain": "", "antigen_chain": chain_id})
        print(f"  {pdb_id}: using antigen chain '{chain_id}'")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="KO-BCE / gKO-BCE conformational B-cell epitope prediction")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        choices=list(config.MODELS),
                        help="Which released model to use (gko-bce = glycosylation-aware flagship)")
    parser.add_argument("--input_pdb_dir", type=str, default=str(config.EXAMPLE_PDB_DIR),
                        help="Directory of antigen (or antibody-antigen complex) PDB files")
    parser.add_argument("--antigen_chain", type=str, default=None,
                        help="Antigen chain ID (default: auto-detect the longest protein chain)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Classification probability threshold (default: %.2f)" % config.TRANSFORMER_THRESHOLD)
    parser.add_argument("--output_dir", type=str, default="predictions",
                        help="Directory where per-residue prediction CSVs are written")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Residues per forward pass. Attention is O(batch * n_features^2); "
                             "raise on large-memory GPUs, lower if you hit OOM.")
    args = parser.parse_args()

    spec = config.MODELS[args.model]
    use_ptms = spec["use_ptms"]
    threshold = args.threshold if args.threshold is not None else config.TRANSFORMER_THRESHOLD

    input_dir = Path(args.input_pdb_dir)
    if not input_dir.exists() or not list(input_dir.glob("*.pdb")):
        print(f"Error: input PDB directory '{input_dir}' is missing or has no .pdb files.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"  {args.model.upper()} B-Cell Epitope Prediction")
    print("=" * 70)
    print(f"  {spec['description']}")
    print(f"  Input PDBs:   {input_dir}")
    print(f"  Glyco feature: {'ENABLED (network lookup of glycosylation sites)' if use_ptms else 'disabled'}")
    print(f"  Threshold:    {threshold}")
    print("=" * 70 + "\n")

    # 1. Fetch weights (downloads on first use, then cached).
    print("Resolving model weights...")
    model_dir = weights.ensure_model(args.model)

    # 2. Isolate antigen chains and build the per-structure index.
    print("\nPreparing antigen-only structures...")
    temp_antigen_dir = config.OUTPUT_DIR / "inference_antigen_only"
    df_pdbs = prepare_antigen_only(input_dir, temp_antigen_dir, args.antigen_chain)
    if df_pdbs.empty:
        print("Error: no usable antigen structures were produced.")
        sys.exit(1)

    # 3. Feature extraction (ESM-2 / ESM-1v / ESM-IF1 + biophysical + optional glyco).
    print("\nExtracting per-residue features...")
    os.makedirs(config.EMBEDDING_CACHE_DIR, exist_ok=True)
    feature_extractor.generate_features(
        df=df_pdbs,
        cleaned_pdb_dir=str(config.OUTPUT_DIR),   # unused when is_training=False
        antigen_only_pdb_dir=str(temp_antigen_dir),
        is_training=False,
        use_ptms=use_ptms,
    )

    # 4. Run the transformer.
    print("\nRunning inference...")
    all_results = transformer_inference.run_inference(
        model_dir=str(model_dir),
        structured_data_path=str(config.STRUCTURED_DATA_PATH),
        threshold=threshold,
        batch_size=args.batch_size,
    )

    # 5. Save per-residue prediction CSVs (sorted by descending probability).
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nSaving predictions...")
    for df in all_results:
        pdb_id = df["pdb_id"].iloc[0]
        csv_path = out_dir / f"{pdb_id}_{args.model}_predictions.csv"
        df.sort_values(by="probability", ascending=False).to_csv(csv_path, index=False)
        n_pos = int((df["prediction"] == 1).sum())
        print(f"  {pdb_id}: {csv_path}  ({n_pos}/{len(df)} predicted epitope residues)")

    print(f"\nDone. Results in '{out_dir}/'.")


if __name__ == "__main__":
    main()
