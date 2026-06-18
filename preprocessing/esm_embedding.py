import os
import numpy as np
import torch
import esm
import esm.inverse_folding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM_IF1_MODEL_NAME = "esm_if1_gvp4_t16_142M_UR50"
ESM1V_MODEL_NAME = "esm1v_t33_650M_UR90S_1"


def get_esm2_embedding(model_tuple, sequence: str) -> torch.Tensor:
    """
    Generates ESM-2 embedding for a single sequence.

    Args:
        model_tuple: A tuple containing the loaded (model, alphabet).
        sequence: The amino acid sequence string.

    Returns:
        A torch tensor of per-residue embeddings.
    """
    model, alphabet = model_tuple
    batch_converter = alphabet.get_batch_converter()
    max_len = 1022
    if len(sequence) > max_len:
        print(f"Sequence truncated to {max_len} residues for ESM-2 embedding.")
        sequence = sequence[:max_len]
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(DEVICE)
    try:
        layer = int(ESM2_MODEL_NAME.split('_')[1][1:])
    except (IndexError, ValueError):
        print(f"Could not parse layer from model name '{ESM2_MODEL_NAME}'. Defaulting to 33.")
        layer = 33 
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer], return_contacts=False) 
    embedding = results["representations"][layer].to("cpu").numpy()
    return torch.tensor(embedding[0, 1 : len(sequence) + 1]).float()


def get_esm1v_embedding(model_tuple, sequence: str) -> torch.Tensor:
    """
    Generates ESM-1v embedding for a single sequence.

    Args:
        model_tuple: A tuple containing the loaded (model, alphabet).
        sequence: The amino acid sequence string.

    Returns:
        A torch tensor of per-residue embeddings.
    """
    model, alphabet = model_tuple
    batch_converter = alphabet.get_batch_converter()
    max_len = 1022
    if len(sequence) > max_len:
        print(f"Sequence truncated to {max_len} residues for ESM-1v embedding.")
        sequence = sequence[:max_len]
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(DEVICE)
    try:
        layer = int(ESM1V_MODEL_NAME.split('_')[1][1:])
    except (IndexError, ValueError):
        print(f"Could not parse layer from model name '{ESM1V_MODEL_NAME}'. Defaulting to 33.")
        layer = 33
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer], return_contacts=False) 
    embedding = results["representations"][layer].to("cpu").numpy()
    return torch.tensor(embedding[0, 1 : len(sequence) + 1]).float()
        

def get_esm_if1_embedding(model_tuple, pdb_path, chain_id) -> torch.Tensor:
    """
    Generates ESM-IF1 3D structure-based embedding for a given chain of a PDB file.
    """
    model, alphabet = model_tuple

    try:
        structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
        if not structure: return None
        coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)
        with torch.no_grad():
            embedding = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
        return torch.tensor(embedding).float()
    except Exception as e:
        print(f"Error getting ESM-IF1 embedding for {os.path.basename(pdb_path)} chain {chain_id}: {e}")
        return None
