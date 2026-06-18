# feature_extractor.py
import os
import pickle
import numpy as np
import pandas as pd
import torch
import warnings
from Bio.PDB import PDBParser, NeighborSearch, Polypeptide, SASA
from Bio.SeqUtils import seq1
from tqdm import tqdm
from sklearn.decomposition import PCA

import config
from preprocessing import esm_embedding as esm_emb
from preprocessing.glycosylation import check_glycosylation

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_amino_acid_one_hot(residue_name):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_map = {aa: i for i, aa in enumerate(amino_acids)}
    try:
        one_letter = seq1(residue_name)
    except KeyError:
        return np.zeros(len(amino_acids), dtype=int)
        
    one_hot = np.zeros(len(amino_acids), dtype=int)
    if one_letter in aa_map:
        one_hot[aa_map[one_letter]] = 1
    return one_hot

def load_esm_models():
    """Loads the specified ESM models into memory."""
    import esm
    models = {}
    if not config.EMBEDDING_MODE:
        print("WARNING: EMBEDDING_MODE is empty. No embedding models will be loaded.")
        return models
    
    if 'esm2' in config.EMBEDDING_MODE:
        print(f"Loading ESM-2 model: {config.ESM2_MODEL_NAME}")
        model, alphabet = esm.pretrained.load_model_and_alphabet(config.ESM2_MODEL_NAME)
        models['esm2'] = (model.to(DEVICE).eval(), alphabet)
    if 'esm_if1' in config.EMBEDDING_MODE:
        print(f"Loading ESM-IF1 model: {config.ESM_IF1_MODEL_NAME}")
        model, alphabet = esm.pretrained.load_model_and_alphabet(config.ESM_IF1_MODEL_NAME)
        models['esm_if1'] = (model.eval(), alphabet)
    if 'esm1v' in config.EMBEDDING_MODE:
        print(f"Loading ESM-1v model: {config.ESM1V_MODEL_NAME}")
        model, alphabet = esm.pretrained.load_model_and_alphabet(config.ESM1V_MODEL_NAME)
        models['esm1v'] = (model.to(DEVICE).eval(), alphabet)
    print(f"ESM models loaded. Running on device: {DEVICE}")
    return models

def get_biophysical_features(structure, antigen_chain_id):
    """Calculates RSA and B-Factor for each residue in the antigen chain."""
    features = {}
    antigen_chain = structure[0][antigen_chain_id]
    
    # Calculate SASA
    sasa_calculator = SASA.ShrakeRupley()
    sasa_calculator.compute(structure, level="R")

    for res in antigen_chain.get_residues():
        if not Polypeptide.is_aa(res, standard=True):
            continue

        res_id_tuple = res.get_id()
        res_name = res.get_resname()
        res_id_str = f"{res_id_tuple[1]}{res_id_tuple[2]}".strip()

        # RSA Calculation
        sasa = res.sasa if hasattr(res, 'sasa') else 0
        max_sasa = config.SASA_MAX_VALUES.get(seq1(res_name), 1.0)
        rsa = sasa / max_sasa if max_sasa > 0 else 0
        
        # B-Factor (average of all atoms in the residue)
        b_factor = np.mean([atom.get_bfactor() for atom in res.get_atoms()])

        features[res_id_str] = {"rsa": rsa, "b_factor": b_factor}
    return features

def get_glycosylation_features(structure, antigen_chain_id, glycosylation_info, use_ptms):
    """Calculates glycosylation features (binary indicator and minimum distance)."""
    if not use_ptms or not glycosylation_info:
        return {}

    antigen_chain = structure[0][antigen_chain_id]
    residues = [res for res in antigen_chain if Polypeptide.is_aa(res, standard=True)]
    
    glycosylation_features = {f"{res.get_id()[1]}{res.get_id()[2]}".strip(): {} for res in residues}
    glycosylated_residue_ids = {int(site['residue_number']) for site in glycosylation_info}
    
    for res in residues:
        res_id_tuple = res.get_id()
        res_id_str = f"{res_id_tuple[1]}{res_id_tuple[2]}".strip()
        is_glycosylated = 1 if res_id_tuple[1] in glycosylated_residue_ids else 0
        glycosylation_features[res_id_str]["is_glycosylated"] = is_glycosylated

    glycosylated_atoms = []
    for site in glycosylation_info:
        try:
            res = antigen_chain[site['residue_number']]
            glycosylated_atoms.extend(list(res.get_atoms()))
        except KeyError:
            continue

    if not glycosylated_atoms:
        for res_id_str in glycosylation_features:
            glycosylation_features[res_id_str]["dist_to_glycosylation"] = config.MAX_GLYCOSYLATION_DISTANCE
    else:
        ns = NeighborSearch(glycosylated_atoms)
        for res in residues:
            min_dist = config.MAX_GLYCOSYLATION_DISTANCE
            if 'CA' in res:
                ca_atom = res['CA']
                nearby_atoms = ns.search(ca_atom.get_coord(), config.MAX_GLYCOSYLATION_DISTANCE, level='A')
                if nearby_atoms:
                    min_dist = min(np.linalg.norm(ca_atom.get_coord() - atom.get_coord()) for atom in nearby_atoms)

            res_id_str = f"{res.get_id()[1]}{res.get_id()[2]}".strip()
            glycosylation_features[res_id_str]["dist_to_glycosylation"] = min_dist
            
    return glycosylation_features

def identify_epitope_residues(structure, h_chain_id, l_chain_id, antigen_chain_id):
    """Identifies epitope residues using a 3D distance threshold between antibody and antigen."""
    model = structure[0]
    antibody_atoms = []
    if h_chain_id in model:
        antibody_atoms.extend(list(model[h_chain_id].get_atoms()))
    if l_chain_id in model:
        antibody_atoms.extend(list(model[l_chain_id].get_atoms()))
    
    if not antibody_atoms:
        return set()

    antigen_residues = [res for res in model[antigen_chain_id] if Polypeptide.is_aa(res)]
    ns = NeighborSearch(antibody_atoms)
    
    # Residue is considered epitope if any of its atoms are within DISTANCE_THRESHOLD of antibody atoms
    epitope_residues = {
        res.get_id() for res in antigen_residues
        if any(ns.search(atom.get_coord(), config.DISTANCE_THRESHOLD, level='A') for atom in res)
    }
    return epitope_residues

def train_and_save_pca(all_embeddings, model_path, target_dim):
    """Fits and saves a PCA model to reduce embedding dimensionality."""
    print(f"  Fitting PCA model for components={target_dim}...")
    stacked_embeddings = np.vstack(all_embeddings)
    pca = PCA(n_components=target_dim)
    pca.fit(stacked_embeddings)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"  PCA components explain {explained_variance:.2%} of original variance.")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pca, f)
    return pca

def generate_features(df, cleaned_pdb_dir, antigen_only_pdb_dir, is_training=True, use_ptms=False, uniprot_id_map=None):
    """
    Main function to process PDB structures and generate features & labels (if training).
    
    Args:
        df (pd.DataFrame): Dataframe listing pdb, Hchain, Lchain, antigen_chain.
        cleaned_pdb_dir (str): Directory containing cleaned complex PDBs (for epitope label resolution).
        antigen_only_pdb_dir (str): Directory containing isolated antigen PDBs (for feature engineering).
        is_training (bool): If True, generates 'is_epitope' labels based on complex structures.
        use_ptms (bool): If True, includes glycosylation PTM features.
        uniprot_id_map (dict): Custom mapping of pdb_id -> UniProt ID.
    """
    print("\n--- Running Feature Extraction Pipeline ---")
    models = load_esm_models()
    parser = PDBParser(QUIET=True)
    
    # 1. Fit or Load PCA Models
    pca_models = {}
    if 'esm2' in config.EMBEDDING_MODE and config.REDUCE_ESM2_DIM:
        pca_model_path = os.path.join(config.PCA_MODEL_CACHE_DIR, f"esm2_pca_{config.ESM2_DIM_TARGET}.pkl")
        if os.path.exists(pca_model_path):
            print(f"  Loading ESM2 PCA from {pca_model_path}")
            with open(pca_model_path, 'rb') as f:
                pca_models['esm2'] = pickle.load(f)
        else:
            # Fit PCA on first-pass sequence embeddings if needed, or initialize a placeholder
            print("  Warning: No pre-fitted ESM2 PCA found. A new one will be fitted during preprocessing.")
            
    if 'esm1v' in config.EMBEDDING_MODE and config.REDUCE_ESM1V_DIM:
        pca_model_path = os.path.join(config.PCA_MODEL_CACHE_DIR, f"esm1v_pca_{config.ESM1V_DIM_TARGET}.pkl")
        if os.path.exists(pca_model_path):
            print(f"  Loading ESM1v PCA from {pca_model_path}")
            with open(pca_model_path, 'rb') as f:
                pca_models['esm1v'] = pickle.load(f)
                
    # We will accumulate unreduced embeddings first if PCA models don't exist
    accumulated_esm2 = []
    accumulated_esm1v = []
    
    # Setup cache directories
    os.makedirs(config.EMBEDDING_CACHE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # First pass: Generate or load raw embeddings
    temp_embeddings = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="First-Pass Embedding Generation"):
        pdb_id = row['pdb']
        antigen_chain_id = row['antigen_chain']
        antigen_only_path = os.path.join(antigen_only_pdb_dir, f"{pdb_id}_antigen_only.pdb")
        
        if not os.path.exists(antigen_only_path):
            # Check for direct pdb input (useful in inference)
            antigen_only_path = os.path.join(antigen_only_pdb_dir, f"{pdb_id}.pdb")
            if not os.path.exists(antigen_only_path):
                continue
                
        try:
            struct = parser.get_structure(pdb_id, antigen_only_path)
            # Find the chain
            chains = list(struct[0].get_chains())
            chain = struct[0][antigen_chain_id] if antigen_chain_id in struct[0] else chains[0]
            residues = [res for res in chain if Polypeptide.is_aa(res, standard=True)]
            if not residues: continue
            
            seq = "".join([seq1(res.get_resname()) for res in residues])
            temp_embeddings[pdb_id] = {'seq': seq, 'antigen_chain_id': chain.id, 'antigen_only_path': antigen_only_path}
            
            if 'esm2' in config.EMBEDDING_MODE:
                esm2_cache_path = os.path.join(config.EMBEDDING_CACHE_DIR, f"{pdb_id}_{chain.id}_esm2.npy")
                if os.path.exists(esm2_cache_path) and not config.FORCE_RECOMPUTE_EMBEDDINGS:
                    emb = np.load(esm2_cache_path)
                else:
                    emb = esm_emb.get_esm2_embedding(models['esm2'], seq).numpy()
                    np.save(esm2_cache_path, emb)
                accumulated_esm2.append(emb)
                
            if 'esm1v' in config.EMBEDDING_MODE:
                esm1v_cache_path = os.path.join(config.EMBEDDING_CACHE_DIR, f"{pdb_id}_{chain.id}_esm1v.npy")
                if os.path.exists(esm1v_cache_path) and not config.FORCE_RECOMPUTE_EMBEDDINGS:
                    emb = np.load(esm1v_cache_path)
                else:
                    emb = esm_emb.get_esm1v_embedding(models['esm1v'], seq).numpy()
                    np.save(esm1v_cache_path, emb)
                accumulated_esm1v.append(emb)
        except Exception as e:
            print(f"  Error on first pass for {pdb_id}: {e}")

    # Fit PCA if they were missing
    if 'esm2' in config.EMBEDDING_MODE and config.REDUCE_ESM2_DIM and 'esm2' not in pca_models and accumulated_esm2:
        pca_model_path = os.path.join(config.PCA_MODEL_CACHE_DIR, f"esm2_pca_{config.ESM2_DIM_TARGET}.pkl")
        pca_models['esm2'] = train_and_save_pca(accumulated_esm2, pca_model_path, config.ESM2_DIM_TARGET)
        
    if 'esm1v' in config.EMBEDDING_MODE and config.REDUCE_ESM1V_DIM and 'esm1v' not in pca_models and accumulated_esm1v:
        pca_model_path = os.path.join(config.PCA_MODEL_CACHE_DIR, f"esm1v_pca_{config.ESM1V_DIM_TARGET}.pkl")
        pca_models['esm1v'] = train_and_save_pca(accumulated_esm1v, pca_model_path, config.ESM1V_DIM_TARGET)

    # Second pass: Extract physical/structural attributes and assemble final matrices
    final_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Assembling Feature Data"):
        pdb_id = row['pdb']
        if pdb_id not in temp_embeddings:
            continue
            
        seq = temp_embeddings[pdb_id]['seq']
        antigen_chain_id = temp_embeddings[pdb_id]['antigen_chain_id']
        antigen_only_path = temp_embeddings[pdb_id]['antigen_only_path']
        
        try:
            struct_only = parser.get_structure(pdb_id, antigen_only_path)
            chains = list(struct_only[0].get_chains())
            chain_only = struct_only[0][antigen_chain_id] if antigen_chain_id in struct_only[0] else chains[0]
            residues = [res for res in chain_only if Polypeptide.is_aa(res, standard=True)]
            
            # Biophysical Features
            biophysical_feats = get_biophysical_features(struct_only, antigen_chain_id)
            
            # Label Resolution (if training)
            epitope_ids = set()
            if is_training:
                complex_path = os.path.join(cleaned_pdb_dir, f"{pdb_id}_cleaned.pdb")
                if os.path.exists(complex_path):
                    struct_complex = parser.get_structure(f"{pdb_id}_complex", complex_path)
                    epitope_ids = identify_epitope_residues(struct_complex, row['Hchain'], row['Lchain'], antigen_chain_id)
            
            # Glycosylation Features
            glycosylation_info = None
            if use_ptms:
                prov_uid = uniprot_id_map.get(pdb_id) if uniprot_id_map else None
                is_glyco, res_nums = check_glycosylation(pdb_id, antigen_chain_id, uniprot_id=prov_uid)
                if is_glyco:
                    glycosylation_info = []
                    for r_num_str in res_nums:
                        try:
                            r_num = int(r_num_str)
                            res = chain_only[r_num]
                            glycosylation_info.append({
                                'chain_id': antigen_chain_id,
                                'residue_number': r_num,
                                'residue_name': res.get_resname()
                            })
                        except KeyError:
                            continue
            
            glycosylation_features = get_glycosylation_features(struct_only, antigen_chain_id, glycosylation_info, use_ptms)
            
            # Embeddings and Dimensionality Reduction
            esm2_embeddings = None
            esm1v_embeddings = None
            esm_if1_embeddings = None
            
            if 'esm2' in config.EMBEDDING_MODE:
                esm2_raw = np.load(os.path.join(config.EMBEDDING_CACHE_DIR, f"{pdb_id}_{antigen_chain_id}_esm2.npy"))
                esm2_embeddings = pca_models['esm2'].transform(esm2_raw) if 'esm2' in pca_models else esm2_raw
                
            if 'esm1v' in config.EMBEDDING_MODE:
                esm1v_raw = np.load(os.path.join(config.EMBEDDING_CACHE_DIR, f"{pdb_id}_{antigen_chain_id}_esm1v.npy"))
                esm1v_embeddings = pca_models['esm1v'].transform(esm1v_raw) if 'esm1v' in pca_models else esm1v_raw
                
            if 'esm_if1' in config.EMBEDDING_MODE:
                esm_if1_embeddings = esm_emb.get_esm_if1_embedding(models['esm_if1'], antigen_only_path, antigen_chain_id)
                if esm_if1_embeddings is not None:
                    esm_if1_embeddings = esm_if1_embeddings.numpy()
                    
            # Assemble Residue Features
            for i, res in enumerate(residues):
                res_id_tuple = res.get_id()
                res_id_str = f"{res_id_tuple[1]}{res_id_tuple[2]}".strip()
                
                # Stack Embeddings
                emb_parts = []
                if 'esm2' in config.EMBEDDING_MODE:
                    if esm2_embeddings is None or i >= len(esm2_embeddings): continue
                    emb_parts.append(esm2_embeddings[i])
                if 'esm1v' in config.EMBEDDING_MODE:
                    if esm1v_embeddings is None or i >= len(esm1v_embeddings): continue
                    emb_parts.append(esm1v_embeddings[i])
                if 'esm_if1' in config.EMBEDDING_MODE:
                    if esm_if1_embeddings is None or i >= len(esm_if1_embeddings): continue
                    emb_parts.append(esm_if1_embeddings[i])
                
                embedding = np.concatenate(emb_parts) if emb_parts else np.zeros(1, dtype=np.float32)
                bio_feats = biophysical_feats.get(res_id_str, {"rsa": 0, "b_factor": 0})
                
                res_dict = {
                    "pdb_id": pdb_id,
                    "antigen_chain": antigen_chain_id,
                    "res_id": res_id_str,
                    "res_name": res.get_resname(),
                    "one_hot_amino_acid": get_amino_acid_one_hot(res.get_resname()),
                    "rsa": bio_feats["rsa"],
                    "b_factor": bio_feats["b_factor"],
                    "seq_length": len(seq),
                    "embedding": embedding,
                }
                
                if is_training:
                    res_dict["is_epitope"] = 1 if res_id_tuple in epitope_ids else 0
                    
                if use_ptms:
                    glyco_feats = glycosylation_features.get(res_id_str, {})
                    res_dict["is_glycosylated"] = glyco_feats.get("is_glycosylated", 0)
                    res_dict["dist_to_glycosylation"] = glyco_feats.get("dist_to_glycosylation", config.MAX_GLYCOSYLATION_DISTANCE)
                    
                final_data.append(res_dict)
                
        except Exception as e:
            print(f"  Error processing features for {pdb_id}: {e}")
            
    if not final_data:
        raise ValueError("No feature data extracted from any PDB structure.")
        
    final_df = pd.DataFrame(final_data)
    final_df.to_pickle(config.FINAL_DATAFRAME_PATH)
    print(f"Saved feature DataFrame of shape {final_df.shape} to {config.FINAL_DATAFRAME_PATH}")
    
    # Process structured dictionaries
    structure_data_to_dict(final_df, is_training, use_ptms)
    return final_df

def structure_data_to_dict(df, is_training, use_ptms):
    """Transforms raw DataFrame into structured lists of protein-specific feature dictionaries."""
    protein_data_list = []
    grouped = df.groupby('pdb_id')
    
    print("Formatting protein groups into dictionary structures...")
    for pdb_id, group in tqdm(grouped, desc="Structuring Proteins"):
        L = len(group)
        embeddings = np.vstack(group['embedding'].values)
        seq_onehot = np.vstack(group['one_hot_amino_acid'].values)
        b_factors = group['b_factor'].values.reshape(-1, 1)
        seq_lengths = group['seq_length'].values.reshape(-1, 1)
        rsas = group['rsa'].values.reshape(-1, 1)

        features = [embeddings, seq_onehot, b_factors, seq_lengths, rsas]

        if use_ptms:
            is_glycosylated = group['is_glycosylated'].values.reshape(-1, 1)
            features.append(is_glycosylated)
            dist_to_glycosylation = group['dist_to_glycosylation'].values.reshape(-1, 1)
            features.append(dist_to_glycosylation)

        X_arr = np.concatenate(features, axis=1)
        embed_dim = embeddings.shape[1]
        
        feature_idxs = {
            "embedding": range(0, embed_dim),
            "sequence_onehot": range(embed_dim, embed_dim + 20),
            "b_factor": range(embed_dim + 20, embed_dim + 21),
            "length": range(embed_dim + 21, embed_dim + 22),
            "rsa": range(embed_dim + 22, embed_dim + 23),
        }
        
        current_idx = embed_dim + 23
        if use_ptms:
            feature_idxs["is_glycosylated"] = range(current_idx, current_idx + 1)
            current_idx += 1
            feature_idxs["dist_to_glycosylation"] = range(current_idx, current_idx + 1)
            current_idx += 1

        df_stats = pd.DataFrame({
            "pdb_id": pdb_id,
            "chain": group['antigen_chain'].values,
            "res_id": group['res_id'].values,
            "residue": group['res_name'].values,
            "rsa": group['rsa'].values,
            "b_factor": group['b_factor'].values,
            "length": group['seq_length'].values,
        })
        
        if is_training:
            df_stats['is_epitope'] = group['is_epitope'].values
            
        if use_ptms:
            df_stats['is_glycosylated'] = group['is_glycosylated'].values
            df_stats['dist_to_glycosylation'] = group['dist_to_glycosylation'].values

        output_dict = {
            "pdb_id": pdb_id,
            "X_arr": X_arr.astype(np.float32),
            "df_stats": df_stats,
            "length": L,
            "feature_idxs": feature_idxs,
        }
        protein_data_list.append(output_dict)
        
    with open(config.STRUCTURED_DATA_PATH, 'wb') as f:
        pickle.dump(protein_data_list, f)
    print(f"Successfully serialized {len(protein_data_list)} structured protein objects to {config.STRUCTURED_DATA_PATH}")
