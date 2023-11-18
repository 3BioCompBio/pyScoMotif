import csv
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Iterator

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from Bio.PDB.QCPSuperimposer import QCPSuperimposer
from joblib import Parallel, delayed

from pyscomotif.data_containers import Residue
from pyscomotif.motif_search import (extract_motif_residues_from_PDB_file,
                                     get_tqdm_progress_bar)
from pyscomotif.utils import (
    detect_the_compression_algorithm_used_in_the_index,
    read_compressed_and_pickled_file)


def get_n_RMSD_to_compute(PDBs_with_similar_motifs:Dict[nx.Graph, Dict[str, List[List[Tuple[str,str]]]]]) -> int:
    """
    """
    n_RMSD_to_compute = sum(
        1 
        for solved_motif_MST, solutions_to_motif in PDBs_with_similar_motifs.items()
            for PDB_ID, list_of_similar_motifs_in_PDB in solutions_to_motif.items()
    )
    return n_RMSD_to_compute

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
        target_motif: Tuple[str, ...], query_motif: Tuple[str, ...], reference_motif_residues_data: Dict[str, Residue], candidate_PDB_residue_data: Dict[str, Residue], RMSD_atoms: str, 
    ) -> float:
    """
    """
    ordered_reference_residue_IDs: List[str] = [full_residue_ID[:-1] for full_residue_ID in query_motif]
    ordered_target_residue_IDs: List[str] = [full_residue_ID[:-1] for full_residue_ID in target_motif]

    reference_coords = get_coordinates_of_motif_residues(reference_motif_residues_data, ordered_reference_residue_IDs, RMSD_atoms)
    target_coords = get_coordinates_of_motif_residues(candidate_PDB_residue_data, ordered_target_residue_IDs, RMSD_atoms)
    
    qcp_superimposer = QCPSuperimposer()

    qcp_superimposer.set(reference_coords, target_coords)
    qcp_superimposer.run()

    RMSD: float = round(qcp_superimposer.get_rms(), ndigits=3)
    
    return RMSD

def calculate_RMSD_values(
        solved_motif_MST: nx.Graph, residue_data_folder_path: Path, compression: str, PDB_ID :str, 
        list_of_similar_motifs_in_PDB: List[List[Tuple[str,str]]], reference_motif_residues_data: Dict[str, Residue], RMSD_atoms: str
    ) -> Tuple[nx.Graph, str, str, Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], float]]:
    """
    """
    candidate_PDB_residue_data = read_compressed_and_pickled_file(residue_data_folder_path / f'{PDB_ID}.{compression}')
    
    similar_motif_RMSD_map: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], float] = {}
    for similar_motif_residue_mapping in list_of_similar_motifs_in_PDB:
        target_motif = tuple(matched_residues[0] for matched_residues in similar_motif_residue_mapping)
        query_motif = tuple(matched_residues[1] for matched_residues in similar_motif_residue_mapping)
        RMSD = calculate_RMSD_between_two_motifs(target_motif, query_motif, reference_motif_residues_data, candidate_PDB_residue_data, RMSD_atoms)
        
        similar_motif_RMSD_map[(target_motif, query_motif)] = RMSD

    header_description: str = candidate_PDB_residue_data.header_description

    return (solved_motif_MST, PDB_ID, header_description, similar_motif_RMSD_map)

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
        motif_MST: nx.Graph, PDB_file: Path, PDBs_with_similar_motifs: Dict[nx.Graph, Dict[str, List[List[Tuple[str,str]]]]], 
        index_folder_path: Path, RMSD_atoms: str, RMSD_threshold: float, n_cores: int, results_output_path: Path, sort_results: bool
    ) -> None:
    """
    ...
    """
    residue_data_folder_path = index_folder_path / 'residue_data_folder'
    compression = detect_the_compression_algorithm_used_in_the_index(index_folder_path)
    reference_motif_residues_data = extract_motif_residues_from_PDB_file(
        PDB_file=PDB_file, 
        motif=tuple(full_residue_ID[:-1] for full_residue_ID in motif_MST.nodes) # Transform MST node identifiers from full residue ID to standard residue ID, e.g: A41G -> A41
    )
    
    tqdm_progress_bar = get_tqdm_progress_bar(total=get_n_RMSD_to_compute(PDBs_with_similar_motifs), desc='RMSD calculations')
    delayed_func: Callable[[nx.Graph, Path, str, str, List[List[Tuple[str,str]]], Dict[str, Residue], str], Tuple[nx.Graph, str, str, Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], float]]] = delayed(calculate_RMSD_values)
    with Parallel(n_jobs=n_cores, return_as='generator') as parallel:
        # Each solved motif has a list of PDBs with a similar motif, and each PDB potentially has more than one similar motif
        results_generator: Iterator[Tuple[nx.Graph, str, str, Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], float]]] = parallel(
            delayed_func(solved_motif_MST, residue_data_folder_path, compression, PDB_ID, list_of_similar_motifs_in_PDB, reference_motif_residues_data, RMSD_atoms)
            for solved_motif_MST, solutions_to_motif in PDBs_with_similar_motifs.items()
                for PDB_ID, list_of_similar_motifs_in_PDB in solutions_to_motif.items()
        )

        with open(results_output_path, 'w', newline='') as file_handle:
            csv_writer = csv.writer(file_handle)
            csv_writer.writerow(['matched_motif','similar_motif_found','RMSD','n_mutations','PDB_ID','header_description']) # Column names

            for solved_motif_MST, PDB_ID, header_description, similar_motif_RMSD_values_map in results_generator:
                for (target_motif, query_motif), RMSD in similar_motif_RMSD_values_map.items(): # Each PDB potentially has more than one similar motif
                    if RMSD > RMSD_threshold:
                        continue

                    n_mutations = calculate_number_of_mutations(motif_MST, solved_motif_MST)

                    row_data: List[str] = [
                        ' '.join(query_motif),
                        ' '.join(target_motif),
                        str(RMSD),
                        str(n_mutations),
                        PDB_ID,
                        header_description,
                    ]
                    csv_writer.writerow(row_data)
                
                tqdm_progress_bar.update()
    

    if sort_results:
        df = pd.read_csv(results_output_path, header=0)
        df.sort_values(by=['n_mutations','RMSD'], ignore_index=True, inplace=True)
        df.to_csv(results_output_path)
        
    return