import json
import os
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

#from scalene import scalene_profiler

import click

from pyscomotif.constants import (INDEX_MAX_ANGLE_VALUE,
                                    INDEX_MAX_DISTANCE_VALUE)
from pyscomotif.indexing import create_index_folder, update_index_folder
from pyscomotif.motif_search import search_index_for_PDBs_with_similar_motifs
from pyscomotif.RMSD_calculation import calculate_RMSD_between_motif_and_similar_motifs


@click.group(help='pyScoMotif: a tool for the discovery of similar 3D structural motifs across proteins.', context_settings={'max_content_width':2000})
def command_line_interface() -> None:
    pass


def check_index_path_option(ctx: Any, param: Any, value: Union[None, Path]) -> Path:
    if value:
        return value
    
    default_index_folder_path: Path = ctx.params['database_path'] / 'pyScoMotif_index'
    # In Click, callbacks run AFTER type casting and its associated validation, so we have to re-run the validations.
    default_index_folder_path = click.Path(file_okay=False, dir_okay=True, readable=True, writable=True, path_type=Path).convert(default_index_folder_path, ctx=ctx, param=param)
    
    return default_index_folder_path

@command_line_interface.command()
@click.argument('database_path', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path))
@click.option('--index_path', default=None, show_default=False, type=click.Path(file_okay=False, dir_okay=True, readable=True, writable=True, path_type=Path), 
              callback=check_index_path_option, help="Full path of the directory that will contain the index files. Defaults to <database_path>/pyScoMotif_index.")
@click.option('--pattern', type=str, default='*.pdb', show_default=True,
               help="File extension pattern of the PDB files in the database that the file detection algorithm should match, including compression. Examples: *.pdb, *.pdb.gz, *.ent , etcetc . Note the use of the '*' wildcard. Also note that only simple unix style patterns are accepted, complex regex patterns will fail. To know if your pattern works, test it with pathlib.Path.rglob.")
@click.option('--compression', type=click.Choice(('bz2', 'gz')), default='bz2', show_default=True,
               help="Compression algorithm to use to compress the index files. This has nothing to do with the compression of the PDB files, for that use the '--pattern' option.")
@click.option('--n_cores', default=1, show_default=True,
               help='Number of cores to use.')
def create_index(database_path: Path, index_path: Path, pattern: str, compression: str, n_cores: int) -> None:
    """
    Command to create the index needed for the rapid search of similar 3D structural motifs with pyScoMotif. If the --index_path option is not provided, a directory 
    named 'pyScoMotif_index' will be created in the given database path. Once the indexing is completed, the directory can be renamed or moved 
    somewhere else. To update an existing index with new PDB files, see the 'update-index' command.

    Note that during the indexing, residues without a carbon alpha atom or without any sidechain heavy atoms (except for Glycine) are ignored.

    \b
    Arguments
    ---------
    DATABASE_PATH  Full path of the directory containing the PDB files to index (e.g: /home/user/Downloads/PDB). The file detection algorithm is recursive, 
    so PDB files in subfolders will also be indexed.
    """
    create_index_folder(database_path, pattern, index_path, compression, n_cores)
    return

@command_line_interface.command()
@click.argument('database_path', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path))
@click.argument('index_path', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True, readable=True, path_type=Path))
@click.option('--pattern', type=str, default='*.pdb',
               help="File extension pattern of the PDB files in the database that the file detection algorithm should match. Examples: *.pdb, *.pdb.gz, *.ent , etcetc . Note the use of the '*' wildcard. Also note that only simple unix style patterns are accepted, complex regex patterns will fail. To know if your pattern works, test it with pathlib.Path.rglob.")
@click.option('--n_cores', default=1, show_default=True,
               help='Number of cores to use.')
def update_index(database_path: Path, pattern: str, index_path: Path, n_cores: int) -> None:
    """
    Command to update an existing pyScoMotif index with new PDB files. The new PDB files are determined by comparing the name 
    of the PDB files with those that have already been indexed, which are stored in the 'indexed_PDB_files.txt' file.

    \b
    Arguments
    ---------
    DATABASE_PATH  Full path of the directory containing the new PDB files to add to the index (e.g: /home/user/Downloads/PDB). The file detection algorithm is 
    recursive, so PDB files in subfolders will also be indexed. Do not worry if the directory contains PDBs that have already been indexed, they will be ignored.
    
    INDEX_PATH  Full path of the pyScoMotif index directory to update (e.g: /home/user/Downloads/PDB/pyScoMotif_index).
    """
    update_index_folder(database_path, pattern, index_path, n_cores)
    return


def check_motif_argument(ctx: Any, param: Any, value: Tuple[str, ...]) -> Tuple[str, ...]:
    if len(value) < 3:
        raise click.BadParameter(f'The motif of interest must have at least 3 residues, but yours had {len(value)}.')
    return value

def check_results_output_path_option(ctx: Any, param: Any, value: Union[None, Path]) -> Path:
    if value:
        if value.suffix != '.csv':
            raise click.BadParameter("The results_output_path option must be a file path ending in '.csv'")
        return value
        
    else:
        now_timestamp = datetime.now().strftime(f'%H%M%S%d%m%Y') # Format = hours+minutes+seconds+day+month+year
        return Path(os.getcwd()) / f'pyScoMotif_result_{now_timestamp}.csv'

def check_residue_type_policy_option(ctx: Any, param: Any, value: str) -> Union[str, Dict[str,List[str]]]:
    if value in ('strict', 'relaxed', 'fully_relaxed'):
        return value

    # The user gives position specific exchanges in a json formated string, we transform it into a python dictionary 
    try:
        loaded_json: Dict[str,str] = json.loads(value) # loads = load string
        position_specific_exchange: Dict[str, List[str]] = {key:list(value) for key, value in loaded_json.items()} # Ex: {'A43':'KRH'} -> {'A43':['K', 'R', 'H']}
    except JSONDecodeError as exception:
        raise exception

    return position_specific_exchange

@command_line_interface.command()
@click.argument('index_path', type=click.Path(exists=True, path_type=Path))
@click.argument('PDB_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path))
@click.argument('motif', nargs=-1, callback=check_motif_argument) # -1 => Unlimited number of arguments
@click.option('--results_output_path', type=click.Path(path_type=Path), default=None, callback=check_results_output_path_option,
              help='Full path of the csv file where the results will be saved (e.g: /home/user/Downloads/similar_motifs.csv). If not given, the results will be saved in the current working directory in a file named with the current timestamp (e.g: pyScoMotif_result_14230214102022.csv)')
@click.option('--residue_type_policy', default='strict', callback=check_residue_type_policy_option,
               help='''Option to control whether mutated versions of the query motif should also be tested. There are 4 possible values: 1) strict (DEFAULT): only similar motifs with strictly identical residues to the query motif are searched. 2) relaxed: finds PDBs with motifs that are similar but with potential mutated residues by allowing residues to mutate according to the residue group they are part of, such as polar, positively charged, aromatic, etcetc (see our Github for the exact grouping of the residues). 3) fully_relaxed: similar to the relaxed mode, but residues can be mutated to any of the 20 possible residues. 4) custom position specific exchange: specify possible residue mutations for the desired positions in your motif using a JSON formated string (e.g: '{"A1":"KL", "A2":"YW", "A5":"E"}'. Note the use of single and double quotes).\n To control the maximum number of simultaneous mutations, see the max_n_mutated_residues option.''')
@click.option('--max_n_mutated_residues', type=click.IntRange(min=1), default=1, show_default=True,
               help='Maximum number of simultaneous mutations that should be allowed when generating mutated motifs through the residue_type_policy option. For example, when set to 1, versions of the query motif with 1 residue mutated according to the given residue_type_policy will be generated and checked, in addition to the reference query motif itself. Note that very high values for this option will result in longer computation time given all the possible combinations of mutated residues will be generated and searched for in the index. For most use cases, we recommend to use a value between 1 and 3.')
@click.option('--distance_delta_thr', type=click.FloatRange(min=0.0, max=INDEX_MAX_DISTANCE_VALUE, min_open=True, max_open=True), default=2.0, show_default=True,
               help='The distance tolerance (in Å) that should be allowed to consider a pair of residues as similar. A higher value will result in more hits with a higher RMSD, and vice versa. Note that very high values will result in higher RAM usage.')
@click.option('--angle_delta_thr', type=click.FloatRange(min=0.0, max=INDEX_MAX_ANGLE_VALUE, min_open=True, max_open=True), default=30.0, show_default=True,
               help='The angle tolerance (in degrees) that should be allowed to consider a pair of residues as similar. A higher value will result in more hits with a higher RMSD, and vice versa. Note that very high values will result in higher RAM usage.')
@click.option('--RMSD_atoms', type=click.Choice(('CA', 'sidechain', 'CA+sidechain')), default='CA+sidechain', show_default=True,
               help="Determines which coordinates are used to calculate the RMSD. CA = carbon alpha atom coordinate, sidechain = average coordinate of the sidechain's heavy atoms, CA+sidechain = both coordinates.")
@click.option('--RMSD_threshold', type=click.FloatRange(min=0, min_open=True), default=1.5, show_default=True, 
               help='Determines the maximum RMSD of the reported similar motifs in the output csv table. To obtain all hits irrespective of RMSD, set this option to a very high value (e.g 10).')
@click.option('--n_cores', default=1, show_default=True,
               help='Number of cores to use. Note that using high distance_delta_thr and/or angle_delta_thr values in conjuction with many cores could result in very high RAM usage.')
def motif_search(
    index_path: Path, pdb_file: Path, motif: Tuple[str, ...], results_output_path: Path, 
    residue_type_policy: Union[str, Dict[str,List[str]]], max_n_mutated_residues: int, distance_delta_thr: float, 
    angle_delta_thr: float, rmsd_atoms: str, rmsd_threshold: float, n_cores: int
    ) -> None:
    """
    Command to perform the search of similar 3D structural motifs.
    
    \b
    Arguments
    ---------
    INDEX_PATH  Full path of the pyScoMotif index directory to use for the motif search (e.g: /home/user/Downloads/PDB/pyScoMotif_index).

    PDB_FILE  Full path of the PDB file containing the motif of interest (e.g: /home/user/Downloads/1A2Y.pdb).

    MOTIF  Space separated residue identifiers corresponding to the motif of interest (e.g: A1 A2 A5 A8 A13).
    """
    motif_MST, PDBs_with_similar_motifs = search_index_for_PDBs_with_similar_motifs(
        index_folder_path=index_path, 
        PDB_file=pdb_file, motif=motif, 
        residue_type_policy=residue_type_policy,
        max_n_mutated_residues=max_n_mutated_residues,
        distance_delta_thr=distance_delta_thr, angle_delta_thr=angle_delta_thr,
        n_cores=n_cores
    )

    pyScoMotif_results_df = calculate_RMSD_between_motif_and_similar_motifs(
        motif_MST=motif_MST, 
        PDB_file=pdb_file, 
        PDBs_with_similar_motifs=PDBs_with_similar_motifs, 
        index_folder_path=index_path,
        RMSD_atoms=rmsd_atoms,
        RMSD_threshold=rmsd_threshold,
        n_cores=n_cores
    )

    pyScoMotif_results_df.to_csv(results_output_path)
    if len(pyScoMotif_results_df) == 0:
        print('No PDB structure with a similar motif was found.')

    return

if __name__ == '__main__':
    #scalene_profiler.start()
    command_line_interface(max_content_width=2000) # max_content_width=2000 allows help texts to span the entire width of the terminal
    #scalene_profiler.stop()
    

# What if the distance is outside of the available distances, say C_alpha_distance of 25 angstroom in an index made with max 20 angstroom ???
# Could stop the solve_motif function earlier by checking if the current list of PDBs with all the pairs is empty, in which case we can stop computing the remaining pairs.
# Currently we don't recalculate the sidechain CMR coordinates of mutated residues in the reference motif, although in theory we should probably do it

