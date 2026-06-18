# sequence_clustering.py
import os
import subprocess
import json
import random
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import config

def extract_sequences_to_fasta(df, antigen_pdb_dir, fasta_path):
    """
    Parses antigen-only PDB files to extract sequences and saves them to a FASTA file.
    The FASTA header is formatted as 'pdb_id|chain_id' for easy parsing later.
    """
    print("\n--- Extracting sequences to FASTA for CD-HIT ---")
    parser = PDBParser(QUIET=True)
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    
    with open(fasta_path, 'w') as f_out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting sequences"):
            pdb_id = row['pdb']
            chain_id = row['antigen_chain']
            pdb_path = os.path.join(antigen_pdb_dir, f"{pdb_id}_antigen_only.pdb")

            if not os.path.exists(pdb_path):
                # Try fallback without '_antigen_only'
                pdb_path = os.path.join(antigen_pdb_dir, f"{pdb_id}.pdb")
                if not os.path.exists(pdb_path):
                    continue

            try:
                structure = parser.get_structure(pdb_id, pdb_path)
                # Find the chain
                chains = list(structure[0].get_chains())
                chain = structure[0][chain_id] if chain_id in structure[0] else chains[0]
                residues = [res for res in chain if is_aa(res, standard=True)]
                sequence = "".join([seq1(res.get_resname()) for res in residues])
                
                if sequence:
                    header = f">{pdb_id}|{chain.id}\n"
                    f_out.write(header)
                    f_out.write(f"{sequence}\n")
            except Exception as e:
                print(f"Warning: Could not process {pdb_path} for sequence. Error: {e}")

def run_cd_hit(fasta_path, output_path, threshold=0.4):
    """
    Runs the CD-HIT command-line tool to cluster sequences.
    """
    print(f"\n--- Running CD-HIT with a {threshold*100:.0f}% identity threshold ---")
    word_size = 2 if 0.4 <= threshold <= 0.5 else 3
    
    cmd = [
        'cd-hit',
        '-i', str(fasta_path),
        '-o', str(output_path),
        '-c', str(threshold),
        '-n', str(word_size),
        '-d', '0'  # Prevents long sequence names from being truncated
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"CD-HIT clustering complete. Output: {output_path}.clstr")
    except FileNotFoundError:
        print("\n*** WARNING: 'cd-hit' command not found. ***")
        print("Please install CD-HIT to perform sequence-based split verification.")
        print("Alternatively, a fallback random group split will be used.")
        raise
    except subprocess.CalledProcessError as e:
        print("\n*** ERROR: CD-HIT failed to run. ***")
        print("Error message:\n", e.stderr)
        raise

def parse_clusters(cluster_file_path):
    """
    Parses the .clstr file from CD-HIT into a dictionary.
    Returns a dict mapping cluster_id -> [pdb_id, pdb_id, ...].
    """
    print(f"\n--- Parsing CD-HIT cluster file: {cluster_file_path} ---")
    clusters = {}
    current_cluster_id = None
    with open(cluster_file_path, 'r') as f:
        for line in f:
            if line.startswith('>Cluster'):
                current_cluster_id = int(line.strip().split()[-1])
                clusters[current_cluster_id] = []
            else:
                pdb_id = line.split('>')[1].split('|')[0]
                if current_cluster_id is not None:
                    clusters[current_cluster_id].append(pdb_id)
    print(f"Found {len(clusters)} sequence clusters.")
    return clusters

def subsample_clusters(clusters, max_size=50):
    """
    Subsamples large clusters to a maximum size to prevent training bias.
    """
    if max_size is None:
        return clusters
    
    print(f"\n--- Subsampling large clusters to a max size of {max_size} ---")
    subsampled_clusters = {}
    total_removed = 0
    for cid, members in clusters.items():
        if len(members) > max_size:
            total_removed += len(members) - max_size
            subsampled_clusters[cid] = random.sample(members, max_size)
        else:
            subsampled_clusters[cid] = members
    
    print(f"Subsampling complete. Removed {total_removed} proteins from oversized clusters.")
    return subsampled_clusters

def create_clustered_splits(clusters, test_size=0.2, val_size=0.1, random_state=42):
    """
    Creates train, validation, and test splits from the cluster dictionary.
    The split is performed on cluster IDs to ensure no homology leakage.
    """
    print("\n--- Creating train/validation/test splits based on clusters ---")
    cluster_ids = list(clusters.keys())
    
    # Split cluster IDs first
    train_val_ids, test_ids = train_test_split(cluster_ids, test_size=test_size, random_state=random_state)
    
    # Calculate proportional validation size
    adjusted_val_size = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=adjusted_val_size, random_state=random_state)
    
    # Map back to protein PDB IDs
    splits = {'train': [], 'val': [], 'test': []}
    for cid in train_ids:
        splits['train'].extend(clusters[cid])
    for cid in val_ids:
        splits['val'].extend(clusters[cid])
    for cid in test_ids:
        splits['test'].extend(clusters[cid])
        
    print(f"Split results:")
    print(f"  Train: {len(splits['train'])} proteins in {len(train_ids)} clusters")
    print(f"  Val: {len(splits['val'])} proteins in {len(val_ids)} clusters")
    print(f"  Test: {len(splits['test'])} proteins in {len(test_ids)} clusters")
    return splits

def generate_splits(df, cdhit_threshold=0.4, max_size=50):
    """Runs the sequence clustering and produces homology-aware train/val/test splits."""
    try:
        extract_sequences_to_fasta(df, config.ANTIGEN_ONLY_PDB_DIR, config.FASTA_PATH)
        run_cd_hit(config.FASTA_PATH, config.CLUSTER_FILE_PATH, cdhit_threshold)
        clusters = parse_clusters(f"{config.CLUSTER_FILE_PATH}.clstr")
        subsampled = subsample_clusters(clusters, max_size)
        splits = create_clustered_splits(subsampled)
        
        with open(config.SPLITS_FILE_PATH, 'w') as f:
            json.dump(splits, f, indent=4)
        print(f"Saved splits to {config.SPLITS_FILE_PATH}")
        return splits
    except Exception as e:
        print(f"CD-HIT split pipeline failed: {e}. Falling back to random split.")
        # Fallback random split
        pdb_ids = df['pdb'].unique().tolist()
        random.seed(42)
        random.shuffle(pdb_ids)
        n = len(pdb_ids)
        test_cnt = int(n * 0.2)
        val_cnt = int(n * 0.1)
        splits = {
            'train': pdb_ids[test_cnt + val_cnt:],
            'val': pdb_ids[test_cnt:test_cnt + val_cnt],
            'test': pdb_ids[:test_cnt]
        }
        with open(config.SPLITS_FILE_PATH, 'w') as f:
            json.dump(splits, f, indent=4)
        print(f"Saved random splits to {config.SPLITS_FILE_PATH}")
        return splits
