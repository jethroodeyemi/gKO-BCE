# structure_cleaning.py
import os
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Residue import Residue
from tqdm import tqdm

class ChainSelect(Select):
    """Helper class to select specific chains from a PDB structure."""
    def __init__(self, chains_to_keep):
        self.chains_to_keep = set(chains_to_keep)

    def accept_chain(self, chain):
        return chain.get_id() in self.chains_to_keep

    def accept_residue(self, residue: Residue) -> int:
        # Filter out heteroatoms, water molecules, etc.
        return 1 if residue.get_id()[0] == ' ' else 0

def clean_pdbs(df, pdb_dir, cleaned_pdb_dir, antigen_only_pdb_dir):
    """
    For each complex listed in the dataframe, produces two output files:
    1. A cleaned complex containing H, L, and Antigen chains for epitope annotation.
    2. An antigen-only file containing ONLY the antigen chain for feature engineering to avoid leakage.
    """
    print("\n--- Cleaning and isolating PDB structures ---")
    parser = PDBParser(QUIET=True)
    io = PDBIO()

    # Ensure output directories exist
    os.makedirs(cleaned_pdb_dir, exist_ok=True)
    os.makedirs(antigen_only_pdb_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning PDBs"):
        pdb_id = row['pdb']
        input_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        complex_output_path = os.path.join(cleaned_pdb_dir, f"{pdb_id}_cleaned.pdb")
        antigen_only_output_path = os.path.join(antigen_only_pdb_dir, f"{pdb_id}_antigen_only.pdb")

        if not os.path.exists(input_path):
            print(f"  Warning: Raw PDB file for {pdb_id} not found at {input_path}. Skipping.")
            continue
        
        if os.path.exists(complex_output_path) and os.path.exists(antigen_only_output_path):
            continue
        
        try:
            structure = parser.get_structure(pdb_id, input_path)
            
            # Save complex with Antibody H+L chains and Antigen chain
            if not os.path.exists(complex_output_path):
                chains_for_complex = [row['Hchain'], row['Lchain'], row['antigen_chain']]
                io.set_structure(structure)
                io.save(complex_output_path, ChainSelect(chains_for_complex))

            # Save Antigen-only chain to prevent antibody feature leak
            if not os.path.exists(antigen_only_output_path):
                chains_for_antigen = [row['antigen_chain']]
                io.set_structure(structure)
                io.save(antigen_only_output_path, ChainSelect(chains_for_antigen))
                   
        except Exception as e:
            print(f"  Error processing PDB {pdb_id}: {e}")
