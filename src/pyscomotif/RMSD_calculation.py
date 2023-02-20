
import concurrent.futures
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from Bio.PDB.QCPSuperimposer import QCPSuperimposer

from pyscomotif.data_containers import Residue
from pyscomotif.motif_search import extract_motif_residues_from_PDB_file
from pyscomotif.utils import (
    detect_the_compression_algorithm_used_in_the_index,
    read_compressed_and_pickled_file)


def get_coordinates_of_motif_residues(residue_data: Dict[str,Residue], ordered_residue_IDs: List[str], RMSD_atoms: str) -> npt.NDArray[np.float64]:
    """
    """
    coordinates: npt.NDArray[np.float64]
    if RMSD_atoms == 'CA':
        coordinates = np.array([residue_data[residue_ID].C_alpha for residue_ID in ordered_residue_IDs])

    elif RMSD_atoms == 'sidechain':
        coordinates = np.array([residue_data[residue_ID].sidechain_CMR for residue_ID in ordered_residue_IDs])

    elif RMSD_atoms == 'CA+sidechain':
        coordinates_list = []
        for residue_ID in ordered_residue_IDs:
            coordinates_list.append(residue_data[residue_ID].C_alpha)
            coordinates_list.append(residue_data[residue_ID].sidechain_CMR)
        
        coordinates = np.array(coordinates_list)

    else:
        raise ValueError(f'{RMSD_atoms} is invalid.')

    return coordinates

def calculate_RMSD_between_two_motifs(
        similar_motif: nx.Graph, reference_motif_residues_data: Dict[str, Residue], similar_motif_residues_data: Dict[str, Residue], RMSD_atoms: str, 
    ) -> float:
    """
    """
    # When performing the sub-graph monomorphism check we saved the residue mapping between the reference motif 
    # and the similar motif because coordinates have to be paired in advance for quaternion based superimposition to work
    residue_mapping_dict: Dict[str,str] = similar_motif.residue_mapping_dict

    ordered_reference_residue_IDs: List[str] = [full_residue_ID[:-1] for full_residue_ID in residue_mapping_dict.values()]
    ordered_target_residue_IDs: List[str] = [full_residue_ID[:-1] for full_residue_ID in residue_mapping_dict.keys()]

    reference_coords = get_coordinates_of_motif_residues(reference_motif_residues_data, ordered_reference_residue_IDs, RMSD_atoms)
    target_coords = get_coordinates_of_motif_residues(similar_motif_residues_data, ordered_target_residue_IDs, RMSD_atoms)
    
    qcp_superimposer = QCPSuperimposer()

    qcp_superimposer.set(reference_coords, target_coords)
    qcp_superimposer.run()

    RMSD: float = round(qcp_superimposer.get_rms(), ndigits=3)
    
    return RMSD

def get_similar_motif_residues_data(residue_data: Dict[str, Residue], nodes: List[str]) -> Dict[str, Residue]:
    """
    Instead of giving to each parallel executor the data of all the residues we only give it the subset needed for the motif. 
    """
    similar_motif_residues_data: Dict[str, Residue] = {}
    for full_residue_ID in nodes:
        residue_ID = full_residue_ID[:-1]
        similar_motif_residues_data[residue_ID] = residue_data[residue_ID]

    return similar_motif_residues_data

def calculate_number_of_mutations(motif_MST: nx.Graph, solved_motif_MST: nx.Graph) -> int:
    """
    ...
    """
    n_mutations: int = 0
    for full_residue_ID in motif_MST.nodes:
        if full_residue_ID not in solved_motif_MST.nodes:
            n_mutations += 1

    return n_mutations

def calculate_RMSD_between_motif_and_similar_motifs(
        motif_MST: nx.Graph, PDB_file: Path, PDBs_with_similar_motifs: Dict[nx.Graph, Dict[str, List[nx.Graph]]], 
        index_folder_path: Path, RMSD_atoms: str, RMSD_threshold: float, n_cores: int
    ) -> pd.DataFrame:
    """
    ...
    """
    residue_data_folder_path = index_folder_path / 'residue_data_folder'
    compression = detect_the_compression_algorithm_used_in_the_index(index_folder_path)
    reference_motif_residues_data = extract_motif_residues_from_PDB_file(
        PDB_file=PDB_file, 
        motif=tuple(full_residue_ID[:-1] for full_residue_ID in motif_MST.nodes) # Transform MST node identifiers from full residue ID to standard residue ID, e.g: A41G -> A41
    )
    
    submitted_futures: Dict[Future[float], Tuple[nx.Graph, str, nx.Graph, str]] = {}
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Each solved motif has a list of PDBs with a similar motif, and each PDB potentially has more than one similar motif -> 3 nested loops
        for solved_motif_MST, solutions_to_motif in PDBs_with_similar_motifs.items():
            for PDB_ID, list_of_similar_motifs_in_PDB in solutions_to_motif.items():
                solution_PDB_residue_data = read_compressed_and_pickled_file(residue_data_folder_path / f'{PDB_ID}.{compression}')
                solution_PDB_header_description: str = solution_PDB_residue_data.header_description
                for similar_motif in list_of_similar_motifs_in_PDB:
                    similar_motif_residues_data = get_similar_motif_residues_data(solution_PDB_residue_data, list(similar_motif.nodes))

                    future = executor.submit(
                        calculate_RMSD_between_two_motifs, 
                        similar_motif, 
                        reference_motif_residues_data, 
                        similar_motif_residues_data, 
                        RMSD_atoms
                    )
                    submitted_futures[future] = (solved_motif_MST, PDB_ID, similar_motif, solution_PDB_header_description)
            
    
    pyScoMotif_results_dict: Dict[str, List[Any]] = {
        'matched_motif':[], 'similar_motif_found':[], 'RMSD':[], 'n_mutations':[], 'PDB_ID':[], 'header_description':[]
    }
    for future in concurrent.futures.as_completed(submitted_futures):
        solved_motif_MST, PDB_ID, similar_motif, header_description = submitted_futures[future]
        residue_mapping_dict: Dict[str,str] = similar_motif.residue_mapping_dict # Mapping between the solved motif's full residue IDs and the full residue IDs of the similar motif found
        n_mutations = calculate_number_of_mutations(motif_MST, solved_motif_MST)
        RMSD = future.result()

        pyScoMotif_results_dict['matched_motif'].append(' '.join(residue_mapping_dict.values()))
        pyScoMotif_results_dict['similar_motif_found'].append(' '.join(residue_mapping_dict.keys()))
        pyScoMotif_results_dict['RMSD'].append(RMSD)
        pyScoMotif_results_dict['n_mutations'].append(n_mutations)
        pyScoMotif_results_dict['PDB_ID'].append(PDB_ID)
        pyScoMotif_results_dict['header_description'].append(header_description)
    
    pyScoMotif_results_df = pd.DataFrame(pyScoMotif_results_dict)
    return pyScoMotif_results_df[pyScoMotif_results_df.RMSD <= RMSD_threshold].sort_values(by=['n_mutations', 'RMSD'], ignore_index=True)
