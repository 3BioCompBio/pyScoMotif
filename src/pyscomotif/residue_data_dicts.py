import gzip
from collections import UserDict
from pathlib import Path
from typing import Dict, Union

import numpy as np
import numpy.typing as npt
from Bio.Data.IUPACData import protein_letters_3to1_extended
protein_letters_3to1_extended = {res_3_letters.upper():res_1_letter for res_3_letters, res_1_letter in protein_letters_3to1_extended.items()} # By default biopython 3 letter codes are cammel case (ie: 'Gly), but they are all upper case (ie: 'GLY') in parsed PDB files
from Bio.PDB import PDBParser

from Bio.PDB.Residue import Residue as Biopython_residue_type
from Bio.PDB.Structure import Structure

from pyscomotif.data_containers import Residue
from pyscomotif.utils import (get_PDB_ID_from_file_path,
                                pickle_and_compress_python_object)


def parse_PDB_with_biopython(PDB_file_path: Path) -> Structure:
    """
    """
    parser = PDBParser(QUIET=True)
    parsed_PDB_file: Structure
    try:    
        if len(PDB_file_path.suffixes) == 1: # PDB file in any of the valid formats (PDB, mmCIF, etc) so Biopython should be able to parse it.
            parsed_PDB_file = parser.get_structure(id='', file=PDB_file_path)
        elif len(PDB_file_path.suffixes) == 2 and PDB_file_path.suffixes[-1] == '.gz':
            with gzip.open(PDB_file_path, 'rt') as decompressed_PDB_file_handle:
                parsed_PDB_file = parser.get_structure(id='', file=decompressed_PDB_file_handle) # The biopython PDB parser also accepts open file handles
        else:
            raise ValueError(f"<{PDB_file_path.suffix}> compressed PDB files are currently not supported, only gunziped (.gz) compressed PDB files are.")

    except Exception as exception:
        raise exception
    
    return parsed_PDB_file

def add_C_alpha_attribute(residue_data: Dict[str, Biopython_residue_type]) -> None:
    """
    """
    residues_to_ignore = []
    for residue_ID, residue in residue_data.items():
        atoms_dict = residue.child_dict
        if not 'CA' in atoms_dict:
            residues_to_ignore.append(residue_ID)
            continue

        residue.C_alpha = atoms_dict['CA'].coord.round(2) # Residue objects are mutable so no need to reasign them to their residue_ID key


    for residue_ID in residues_to_ignore:
        del residue_data[residue_ID]

    return

def add_sidechain_CMR_attribute(residue_data: Dict[str, Biopython_residue_type]) -> None:
    """
    """
    residues_to_ignore = []
    for residue_ID, residue in residue_data.items():
        if residue.resname == 'GLY':
            sidechain_atoms_coords = np.array([residue.C_alpha])
        else:
            atoms_list = residue.child_list
            sidechain_atoms_coords = np.array([
                atom.coord
                for atom in atoms_list
                if atom.name not in {'N', 'CA', 'C', 'O'} and atom.element != 'H' # All atoms except main chain and hydrogens
            ])
        
        if len(sidechain_atoms_coords) == 0:
            residues_to_ignore.append(residue_ID)
            continue

        residue.sidechain_CMR = sidechain_atoms_coords.mean(axis=0).round(2)


    for residue_ID in residues_to_ignore:
        del residue_data[residue_ID]

    return

def calculate_glycine_vector(residue: Biopython_residue_type) -> Union[npt.NDArray[np.float64], None]:
    """
    Given we cannot calculate a vector between the C_alpha and the side chain CMR as there is no sidechain
    coordinate to be used for Glycine, we calculate a vector that points towards the direction of the C_alpha
    by taking the mid-point between the N and C coordinates, which is simply the average of the x/y/z
    coordinates of the two points, and use that to calculate the vector mid-point -> C_alpha.
    """
    atoms_dict = residue.child_dict
    if not ('N' in atoms_dict and 'C' in atoms_dict):
        return None
    
    N_coord: npt.NDArray[np.float64] = atoms_dict['N'].coord
    C_coord: npt.NDArray[np.float64] = atoms_dict['C'].coord
    C_alpha: npt.NDArray[np.float64] = residue.C_alpha

    N_C_midpoint: npt.NDArray[np.float64] = np.array([N_coord, C_coord]).mean(axis=0)
    return C_alpha - N_C_midpoint

def add_C_alpha_sidechain_CMR_vector_attribute(residue_data: Dict[str, Biopython_residue_type]) -> None:
    """
    """
    residues_to_ignore = []
    for residue_ID, residue in residue_data.items():
        vector = residue.sidechain_CMR - residue.C_alpha if residue.resname != 'GLY' else calculate_glycine_vector(residue)
        if vector is None:
            residues_to_ignore.append(residue_ID)
            continue

        residue.vector = vector.round(2)


    for residue_ID in residues_to_ignore:
        del residue_data[residue_ID]

    return

def add_PDB_header_description_as_dict_attribute(residue_data: UserDict[str,Residue], parsed_PDB_file: Structure) -> None:
    """
    """
    header_description: str = parsed_PDB_file.header['name']

    residue_data.header_description = header_description #type: ignore
    
    return

def extract_residue_data(PDB_file_path: Path) -> UserDict[str, Residue]:
    """
    ...
    """
    parsed_PDB_file = parse_PDB_with_biopython(PDB_file_path)

    residue_data = {
        f'{chain.id}{biopython_residue_object.id[1]}':biopython_residue_object # Key example: A43.
        for chain in parsed_PDB_file.get_chains()
            for biopython_residue_object in chain
    }

    add_C_alpha_attribute(residue_data)
    add_sidechain_CMR_attribute(residue_data)
    add_C_alpha_sidechain_CMR_vector_attribute(residue_data)

    # The Biopython Residue objects contain a lot of data we are not interested in, so we replace them with our own Residue object 
    # that only contains the data we actually care about. 
    # We also want to have access to the PDB's header description/title, which is why we use a UserDict which allows us to add that 
    # string as an attribute to the UserDict object instead of a key in the dictionary, which would be confusing as it's not a Residue object.
    customized_residue_data = UserDict({
        residue_id:Residue(resname=protein_letters_3to1_extended[residue.resname], C_alpha=residue.C_alpha, sidechain_CMR=residue.sidechain_CMR, vector=residue.vector)
        for residue_id, residue in residue_data.items()
        if residue.resname in protein_letters_3to1_extended
    })
    add_PDB_header_description_as_dict_attribute(customized_residue_data, parsed_PDB_file)
    
    return customized_residue_data

def get_compressed_residue_data_output_file_path(PDB_file_path: Path, index_folder_path: Path, compression: str) -> Path:
    """
    """
    PDB_ID = get_PDB_ID_from_file_path(PDB_file_path)
    residue_data_folder_path = index_folder_path / 'residue_data_folder'
    compressed_residue_data_output_file_path = residue_data_folder_path / f'{PDB_ID}.{compression}' # Ex: /home/user_name/database_folder/pyScoMotif_index/residue_data_folder/1A2Y.bz2

    return compressed_residue_data_output_file_path

def generate_compressed_residue_data_from_PDB_file(PDB_file_path: Path, index_folder_path: Path, compression: str) -> None:
    """
    """
    residue_data = extract_residue_data(PDB_file_path)
    pickle_and_compress_python_object(
        python_object=residue_data,
        output_file_path=get_compressed_residue_data_output_file_path(PDB_file_path, index_folder_path, compression)
    )

    return
