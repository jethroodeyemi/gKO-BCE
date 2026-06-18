import requests
import time
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple

# --- API Endpoints ---
PDB_GRAPHQL_API = "https://data.rcsb.org/graphql"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{}.json"


def build_graphql_query(pdb_ids: List[str]) -> str:
    formatted_pdb_ids = ", ".join(f'"{id}"' for id in pdb_ids)
    query = f"""
    {{
      entries(entry_ids: [{formatted_pdb_ids}]) {{
        rcsb_id
        polymer_entities {{
          rcsb_polymer_entity_container_identifiers {{
            reference_sequence_identifiers {{
              database_accession
              database_name
            }}
          }}
          polymer_entity_instances {{
            rcsb_polymer_entity_instance_container_identifiers {{
              asym_id
              auth_asym_id
            }}
          }}
        }}
      }}
    }}
    """
    return query


def fetch_pdb_data(pdb_id: str) -> dict:
    pdb_data_map = {}
    query = build_graphql_query([pdb_id])
    try:
        response = requests.post(PDB_GRAPHQL_API, json={'query': query})
        response.raise_for_status()
        data = response.json()
        for entry in data.get("data", {}).get("entries", []):
            pdb_data_map[entry['rcsb_id']] = entry['polymer_entities']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching PDB data for {pdb_id}: {e}")

    return pdb_data_map


def get_uniprot_id(entity_data: list, chain_id_from_file: str) -> Optional[str]:
    """
    Parses the polymer entity data from GraphQL to find the correct UniProt ID
    for a given chain ID. This version is robust against missing data keys.
    """
    for entity in entity_data:
        for instance in entity.get('polymer_entity_instances', []):
            ids = instance.get(
                'rcsb_polymer_entity_instance_container_identifiers', {})

            if ids.get('asym_id') == chain_id_from_file or ids.get('auth_asym_id') == chain_id_from_file:
                # This is the correct entity. Now safely find its UniProt ID.

                # Step 1: Safely get the container dictionary
                identifiers_container = entity.get(
                    'rcsb_polymer_entity_container_identifiers')

                # Step 2: Check if the container itself exists before proceeding
                if identifiers_container:
                    # Step 3: Get the list of reference sequences. This could be None.
                    ref_seqs = identifiers_container.get(
                        'reference_sequence_identifiers')

                    # Step 4: CRITICAL FIX - Check if ref_seqs is a list before looping
                    if ref_seqs:
                        for ref in ref_seqs:
                            if ref.get('database_name') == 'UniProt':
                                return ref.get('database_accession')

                # If we get here, no UniProt ID was found for this matching chain
                return None

    return None  # Return None if the chain itself was never found


def get_glycosylation_sites(uniprot_id: str) -> List[Dict]:
    if not uniprot_id:
        return []
    url = UNIPROT_API.format(uniprot_id)
    glycosylation_info = []
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for feature in data.get("features", []):
                if feature.get("type") == "Glycosylation":
                    location = feature.get('location', {})
                    position = location.get('start', {}).get('value', 'N/A')
                    description = feature.get("description", "No description")
                    full_description_str = f"Residue {position}: {description}"
                    glycosylation_info.append({
                        "position": str(position),
                        "description": full_description_str
                    })
            return glycosylation_info
        else:
            return []
    except requests.exceptions.RequestException:
        return []


def is_valid_pdb_id(pdb_id: str) -> bool:
    """
    Checks if a PDB ID is valid by querying the RCSB PDB database.
    
    Args:
        pdb_id (str): The PDB ID to validate.
        
    Returns:
        bool: True if the PDB ID exists in RCSB, False otherwise.
    """
    if not pdb_id or len(pdb_id) != 4:
        return False
    
    pdb_data_map = fetch_pdb_data(pdb_id)
    return pdb_id.upper() in pdb_data_map


def check_glycosylation(pdb_id: str, chain_id: str, uniprot_id: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    Checks if a protein chain is glycosylated and returns the sites.

    Args:
        pdb_id (str): The PDB ID of the protein.
        chain_id (str): The chain ID of the protein.
        uniprot_id (str, optional): If provided, use this UniProt ID directly.

    Returns:
        tuple[bool, list[str]]: A tuple containing a boolean indicating if the protein
                                 is glycosylated and a list of glycosylation residue numbers.
    """
    if uniprot_id:
        print(f"  Using provided UniProt ID: {uniprot_id}")
        time.sleep(0.05)  # Be polite to the API
        glycosylation_sites = get_glycosylation_sites(uniprot_id)
        
        if glycosylation_sites:
            residue_numbers = [site['position']
                               for site in glycosylation_sites if site['position'] != 'N/A']
            return True, residue_numbers
        else:
            return False, []
    
    # Otherwise, try to get UniProt ID from PDB ID
    pdb_data_map = fetch_pdb_data(pdb_id)
    entity_data = pdb_data_map.get(pdb_id.upper())
    if not entity_data:
        print(f"  Warning: Could not fetch PDB data for '{pdb_id}'. PTM features will be disabled for this file.")
        return False, []

    resolved_uniprot_id = get_uniprot_id(entity_data, chain_id)
    if not resolved_uniprot_id:
        print(f"  Warning: Could not find UniProt ID for PDB '{pdb_id}' chain '{chain_id}'. PTM features will be disabled.")
        return False, []

    time.sleep(0.05)  # Be polite to the API
    glycosylation_sites = get_glycosylation_sites(resolved_uniprot_id)

    if glycosylation_sites:
        residue_numbers = [site['position']
                           for site in glycosylation_sites if site['position'] != 'N/A']
        return True, residue_numbers
    else:
        return False, []
