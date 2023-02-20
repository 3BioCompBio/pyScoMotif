import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple
from typing_extensions import TypeAlias

import pandas as pd

from pyscomotif.constants import (INDEX_ANGLE_BIN_SIZE,
                                    INDEX_DISTANCE_BIN_SIZE,
                                    INDEX_MAX_DISTANCE_VALUE)
from pyscomotif.data_containers import Residue
from pyscomotif.index_folders_and_files import create_index_folder_tree
from pyscomotif.residue_data_dicts import \
    generate_compressed_residue_data_from_PDB_file
from pyscomotif.utils import (
    detect_the_compression_algorithm_used_in_the_index,
    generate_all_pairs_of_residues_combinations, get_bin_number,
    get_C_alpha_distance, get_PDB_ID_from_file_path,
    get_sidechain_CMR_distance, get_vector_angle,
    pickle_and_compress_python_object, read_compressed_and_pickled_file)


def get_geometric_attribute(geometric_descriptor: str) -> str:
    return geometric_descriptor.rsplit('_', maxsplit=1)[0] # Ex: C_alpha_distance -> C_alpha

def get_residue_specific_data(residue_data: Dict[str, Residue], residue_name: str) -> Dict[str, Residue]:
    """
    ...
    """
    return {residue_ID:residue for residue_ID, residue in residue_data.items() if residue.resname == residue_name}

index_data_type_alias: TypeAlias = List[Tuple[float, float, float, str, str, str]] # For mypy annotation
def get_index_data(residue_data_of_residue_1: Dict[str, Residue], residue_data_of_residue_2: Dict[str, Residue], PDB_ID: str) -> index_data_type_alias:
    """
    ...
    """
    indexed_residue_pairs: Set[str] = set()
    index_data: index_data_type_alias = []
    for residue_1_ID, residue_1 in residue_data_of_residue_1.items():
        for residue_2_ID, residue_2 in residue_data_of_residue_2.items():
            # In the particular case of pairs of identical residue types (e.g: AA, CC, etc), we have to take care of two edge cases:
            # - 1) Avoid including a residue pair of a residue with itself (e.g (A1C, A1C))
            # - 2) Avoid including twice the same residue pair (e.g (A1C, A5C) and (A5C, A1C)) as they are the same
            if residue_1.resname == residue_2.resname:
                if residue_1_ID == residue_2_ID:
                    continue
                if residue_2_ID+residue_1_ID in indexed_residue_pairs:
                    continue
            
            C_alpha_distance = get_C_alpha_distance(residue_1, residue_2)
            if C_alpha_distance > INDEX_MAX_DISTANCE_VALUE:
                continue

            sidechain_CMR_distance = get_sidechain_CMR_distance(residue_1, residue_2)
            vector_angle = get_vector_angle(residue_1, residue_2)

            index_data.append((round(C_alpha_distance, 2), round(sidechain_CMR_distance, 2), round(vector_angle, 2), residue_1_ID, residue_2_ID, PDB_ID))
            indexed_residue_pairs.add(residue_1_ID+residue_2_ID)

    return index_data

def generate_index_data(residue_data: Dict[str, Residue], PDB_ID:str, residue_pair: str) -> index_data_type_alias:
    """
    ...
    """
    residue_1_name, residue_2_name = residue_pair[0], residue_pair[1]
    
    residue_data_of_residue_1 = get_residue_specific_data(residue_data, residue_1_name)
    residue_data_of_residue_2 = get_residue_specific_data(residue_data, residue_2_name)

    index_data = get_index_data(residue_data_of_residue_1, residue_data_of_residue_2, PDB_ID)

    return index_data

def write_index_data_to_corresponding_bin_files(cumulated_index_data: index_data_type_alias, residue_pair: str, index_folder: Path) -> None:
    """
    ...
    """
    # Residues are written to files corresponding to the combination of the 3 bin values for their geometric descriptors (e.g: AG_4_4_5).
    # The bin is determined by simply calculating the floor division between the metric value and the bin size. Note that the use of floor division 
    # leads to half open intervals.
    # Distance bin example: value = 2.3 -> 2.3 //  1 = 2 = bin number
    # Angle bin example   : value = 160 -> 160 // 20 = 8 = bin number
    for row in cumulated_index_data:
        C_alpha_distance, sidechain_CMR_distance, vector_angle_bin = row[0], row[1], row[2]

        C_alpha_distance_bin = get_bin_number(C_alpha_distance, INDEX_DISTANCE_BIN_SIZE)
        sidechain_CMR_distance_bin = get_bin_number(sidechain_CMR_distance, INDEX_DISTANCE_BIN_SIZE)
        vector_angle_bin = get_bin_number(vector_angle_bin, INDEX_ANGLE_BIN_SIZE)

        with open(index_folder / f'{residue_pair}_{C_alpha_distance_bin}_{sidechain_CMR_distance_bin}_{vector_angle_bin}.csv', 'a') as file_handle:
            file_handle.write(','.join((str(element) for element in row)) + '\n')

    return

def create_index_files_of_the_residue_pair_combination(residue_pair: str, PDB_files_to_index: List[Path], index_folder_path: Path, compression: str) -> None:
    """
    ...
    """
    compressed_residue_data_folder = index_folder_path / 'residue_data_folder'
    index_folder = index_folder_path / 'index'

    cumulated_index_data: index_data_type_alias = []
    for PDB_file in PDB_files_to_index:
        PDB_ID = get_PDB_ID_from_file_path(PDB_file)
        compressed_residue_data_file_path = compressed_residue_data_folder / f'{PDB_ID}.{compression}' 
        residue_data: Dict[str, Residue] = read_compressed_and_pickled_file(compressed_residue_data_file_path)
        
        index_data = generate_index_data(residue_data, PDB_ID, residue_pair)

        cumulated_index_data.extend(index_data)
        
        # To limit RAM usage even when dealing with hundreds of thousands of PDB files, the index data is periodically written
        # to the corresponding csv files on disk.
        if len(cumulated_index_data) >= 1000:
            write_index_data_to_corresponding_bin_files(cumulated_index_data, residue_pair, index_folder)
            cumulated_index_data.clear()

    # The above condition never triggers for the last chunk of data, so we need to call the functions one last time.
    write_index_data_to_corresponding_bin_files(cumulated_index_data, residue_pair, index_folder)
    cumulated_index_data.clear()

    # Replace each csv file with a pickled and compressed pandas dataframe. If the pickled and compressed file already
    # exists that means we are in update mode, so we simply read the existing file and concatenate the two dataframes.
    for binned_data_csv_file_path in index_folder.glob(f'{residue_pair}_*.csv'):
        bin_index_data_df = pd.read_csv(
            binned_data_csv_file_path, 
            names=['C_alpha_distance', 'sidechain_CMR_distance', 'vector_angle', 'residue_1','residue_2','PDB_ID'], 
            dtype={'C_alpha_distance':float,'sidechain_CMR_distance':float, 'vector_angle':float, 'residue_1:':str, 'residue_2':str, 'PDB_ID':str},
            engine='c'
        )
        output_file_path=binned_data_csv_file_path.with_suffix(f'.{compression}') # Ex: /home/user_name/database_folder/pyScoMotif_index/index/AG_4_4_5.bz2
        if output_file_path.exists(): 
            # Update mode
            bin_index_data_df = pd.concat((bin_index_data_df, read_compressed_and_pickled_file(output_file_path)), axis=0, copy=False) 
        
        # Replace the csv file with the pickled and compressed pandas dataframe
        binned_data_csv_file_path.unlink()
        pickle_and_compress_python_object(
            python_object=bin_index_data_df,
            output_file_path=output_file_path
        )

    return

def update_file_of_indexed_PDB_files(index_folder: Path, new_PDB_IDs: Iterator[str]) -> None:
    """
    ...
    """
    with open(index_folder / 'indexed_PDB_files.txt', 'a') as file_handle:
        for PDB_ID in new_PDB_IDs:
            file_handle.write(PDB_ID + '\n')
    return

def index_PDB_files(PDB_files_to_index: List[Path], index_folder_path: Path, compression: str, n_cores: int) -> None:
    """
    """
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # We first parse all the PDB files to extract and format the residue data of interest (residue name, C alpha coordinates, etc), 
        # which is then saved as a pickle and compressed file. This is done to avoid parsing anew each PDB file 210 times, instead 
        # we just load these files which contain a python dictionary, which is ~15x faster.
        submited_futures = [executor.submit(generate_compressed_residue_data_from_PDB_file, PDB_file_path, index_folder_path, compression) for PDB_file_path in PDB_files_to_index]
        for future in concurrent.futures.as_completed(submited_futures):
            if future.exception():
                raise future.exception() # type: ignore 
        
        # Each worker processes 1 of the 210 residue pair combinations. The output is a series of pickled and compressed files for each residue pair that contain 
        # the data of a bin e.g .../pyScoMotif_index/index/AG_4_4_5.bz2 contains the occurences of all AG pairs with a C alpha distance between [4, 5) angstrooms, 
        # a CMR distance between [4, 5) and a vector angle between [100, 120) degrees.
        submited_futures = [
            executor.submit(create_index_files_of_the_residue_pair_combination, combination, PDB_files_to_index, index_folder_path, compression) 
            for combination in generate_all_pairs_of_residues_combinations()
        ]
        for future in concurrent.futures.as_completed(submited_futures):
            if future.exception():
                raise future.exception() # type: ignore


    # To allow users to update the index with new structures we need to keep track of the PDB files that have already been indexed.
    update_file_of_indexed_PDB_files(
        index_folder_path, 
        new_PDB_IDs = (get_PDB_ID_from_file_path(PDB_file_path) for PDB_file_path in PDB_files_to_index)
    )
    
    return

def create_index_folder(database_path: Path, pattern: str, index_folder_path: Path, compression: str, n_cores: int) -> None:
    """Indexes all the pairs of residues in the PDB files of the given database."""
    PDB_files_to_index: List[Path] = list(database_path.rglob(pattern)) # rglob = recursively glob through the directory and subdirectory
    if not PDB_files_to_index:
        raise ValueError(f"No files were found that match the pattern {pattern}.")
    
    create_index_folder_tree(index_folder_path)

    index_PDB_files(PDB_files_to_index, index_folder_path, compression, n_cores)
    return

def get_set_of_already_indexed_PDBs(index_folder_path: Path) -> Set[str]:
    """
    Returns the set of PDBs that have already been added to the index.
    """
    with open(index_folder_path / 'indexed_PDB_files.txt', 'rt') as file_handle:
        set_of_already_indexed_PDBs = set(PDB_ID.strip() for PDB_ID in file_handle.readlines())

    return set_of_already_indexed_PDBs

def update_index_folder(database_path: Path, pattern: str, index_folder_path: Path, n_cores: int) -> None:
    """Updates an existing index with new PDB files."""
    set_of_already_indexed_PDBs = get_set_of_already_indexed_PDBs(index_folder_path)
    PDB_files_to_index: List[Path] = []
    for PDB_file in database_path.rglob(pattern):
        PDB_ID = get_PDB_ID_from_file_path(PDB_file)
        if PDB_ID not in set_of_already_indexed_PDBs:
            PDB_files_to_index.append(PDB_file)
    del set_of_already_indexed_PDBs # No longer needed
    
    if not PDB_files_to_index:
        raise ValueError(f"No new files were found that match the pattern {pattern}.")

    compression = detect_the_compression_algorithm_used_in_the_index(index_folder_path)

    index_PDB_files(PDB_files_to_index, index_folder_path, compression, n_cores)

    print(f'{len(PDB_files_to_index)} new structures have been added to the index.')
    return
