import bz2
import gzip
import itertools
import pickle
from pathlib import Path
from typing import Any, List, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt

from pyscomotif.constants import AMINO_ACID_ALPHABET
from pyscomotif.data_containers import Residue


def get_PDB_ID_from_file_path(PDB_file_path: Path) -> str:
    return PDB_file_path.name.split('.')[0]

def get_sorted_2_tuple(tuple_: Tuple[Any, Any]) -> Tuple[Any, Any]:
    return tuple_ if tuple_[0] <= tuple_[1] else (tuple_[1], tuple_[0])

def sort_and_join_2_strings(str1: str, str2: str) -> str:
    return str1+str2 if str1 <= str2 else str2+str1


def pickle_and_compress_python_object(python_object: Any, output_file_path: Path) -> None:
    """
    ...
    """
    if output_file_path.suffix == '.bz2':
        with bz2.open(output_file_path, 'wb') as file_handle:
            pickle.dump(python_object, file_handle)
    elif output_file_path.suffix == '.gz':
        with gzip.open(output_file_path, 'wb') as file_handle:
            pickle.dump(python_object, file_handle)
    else:
        raise ValueError(f"'{output_file_path.suffix}' is currently not supported, only 'bz2' (default) and 'gz' compression are.")

    return

def read_compressed_and_pickled_file(file_path: Path) -> Any:
    """
    """
    python_object: Any
    if file_path.suffix == '.bz2':
        with bz2.open(file_path, 'rb') as file_handle:
            python_object = pickle.load(file_handle)
    elif file_path.suffix == '.gz':
        with gzip.open(file_path, 'rb') as file_handle:
            python_object = pickle.load(file_handle)
    else:
        raise ValueError(f"'{file_path.suffix}' is currently not supported, only 'bz2' (default) and 'gz' compression are.")

    return python_object

def detect_the_compression_algorithm_used_in_the_index(index_folder_path: Path) -> str:
    """
    Detection of the compression algorithm is simply done by taking a file from the residue data folder and checking its extension.
    """
    residue_data_folder = index_folder_path / 'residue_data_folder'
    random_file = next(residue_data_folder.glob('*'))

    compression = random_file.suffix[1:] # We don't want to include the leading dot (.) of the suffix. Ex: '.txt' -> 'txt'
    return compression

def generate_all_pairs_of_residues_combinations() -> List[str]:
    """
    Returns a list of strings that correspond to all the possible pairs of amino acids, in alphabetical order (e.g: AA, AC, AD, ..., YY)
    """
    return [AA1+AA2 for AA1, AA2 in itertools.combinations_with_replacement(sorted(AMINO_ACID_ALPHABET), 2)] # with_replacement needed so that we also get AA, CC, DD, etcetc


@numba.njit() # type: ignore
def get_bin_number(metric_value: float, bin_size: float) -> int:
    return int(metric_value // bin_size)

@numba.njit() # type: ignore
def pairwise_euclidean_distance(v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64]) -> Union[np.float64, float]:
    return np.sqrt(np.sum((v1 - v2)**2)) # type: ignore

@numba.njit() # type: ignore
def angle_between_two_vectors(v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64]) -> Union[np.float64, float]:
    """
    Returns the angle between the two vectors in degrees (°). 
    Largely taken from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    v1_u, v2_u = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    
    dot_product = np.dot(v1_u, v2_u)
    # Clip the value to between -1 and 1 (numba does not support np.clip)
    if dot_product > 1:
        dot_product = 1
    elif dot_product < -1:
        dot_product = -1

    return np.rad2deg(np.arccos(dot_product)) # type: ignore

def get_C_alpha_distance(residue_1: Residue, residue_2: Residue) -> float:
    return float(pairwise_euclidean_distance(residue_1.C_alpha, residue_2.C_alpha))

def get_sidechain_CMR_distance(residue_1: Residue, residue_2: Residue) -> float:
    return float(pairwise_euclidean_distance(residue_1.sidechain_CMR, residue_2.sidechain_CMR))

def get_vector_angle(residue_1: Residue, residue_2: Residue) -> float:
    return float(angle_between_two_vectors(residue_1.vector, residue_2.vector))
