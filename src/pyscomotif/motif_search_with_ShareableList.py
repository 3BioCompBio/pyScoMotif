# This version of the motif search code attemps to speed up the search when dealing with many mutated motifs. The idea is that when 
# dealing with many mutated versions of a reference motif, the data of the same residue pairs is generally loaded over and over again, 
# in particular the residue pair data of the reference motif. The optimization is to therefore load the data of those residue pairs
# once and share it between the parallel process, which can be done using a ShareableList.
# There are two problems with this implementation that need to be addresses before the code can be relased:
# 1) There is a warning that says the shared_memory object is leaked and was not properly cleaned up, despite running shm.close() and shm.unlink() in the code
# 2) The ShareableList documentation says that all items in the list must be < 10M bytes, so the code might crash if there is a residue pair that has a lot of data.

import itertools
import pickle
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import ShareableList
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple, Union

import networkx as nx
import pandas as pd

from pyscomotif.constants import (AMINO_ACID_ALPHABET,
                                    AMINO_ACID_RELAXED_GROUPS_MAP,
                                    INDEX_ANGLE_BIN_SIZE,
                                    INDEX_DISTANCE_BIN_SIZE)
from pyscomotif.data_containers import Residue, Residue_pair_data
from pyscomotif.index_folders_and_files import index_folder_exists
from pyscomotif.residue_data_dicts import extract_residue_data
from pyscomotif.utils import (
    angle_between_two_vectors,
    detect_the_compression_algorithm_used_in_the_index, get_bin_number,
    get_sorted_2_tuple, pairwise_euclidean_distance,
    read_compressed_and_pickled_file, sort_and_join_2_strings)


def extract_motif_residues_from_PDB_file(PDB_file: Path, motif: Tuple[str,...]) -> Dict[str, Residue]:
    """
    ...
    """
    residue_data = extract_residue_data(PDB_file)

    motif_residues_data: Dict[str, Residue] = {}
    for residue_ID in motif:
        if residue_ID not in residue_data:
            raise ValueError(f"Could not find residue '{residue_ID}' in PDB_file {str(PDB_file)}.")
        
        motif_residues_data[residue_ID] = residue_data[residue_ID]

    return motif_residues_data

def get_full_residue_ID(residue_ID: str, resname: str) -> str:
    return residue_ID + resname

def get_data_of_all_pairs_of_residues_in_motif(motif_residues_data: Dict[str, Residue]) -> Dict[Tuple[str,str], Residue_pair_data]:
    """
    ...
    """
    data_of_all_pairs_of_residues_in_motif: Dict[Tuple[str,str], Residue_pair_data] = {}
    for (residue_1_ID, residue_1), (residue_2_ID, residue_2) in itertools.combinations(motif_residues_data.items(), 2):
        residue_1_full_ID, residue_2_full_ID = get_full_residue_ID(residue_1_ID, residue_1.resname), get_full_residue_ID(residue_2_ID, residue_2.resname)
        pair_of_residues_full_IDs: Tuple[str,str] = (residue_1_full_ID, residue_2_full_ID) 

        data_of_all_pairs_of_residues_in_motif[pair_of_residues_full_IDs] = Residue_pair_data(
            C_alpha_distance=pairwise_euclidean_distance(residue_1.C_alpha, residue_2.C_alpha), 
            sidechain_CMR_distance=pairwise_euclidean_distance(residue_1.sidechain_CMR, residue_2.sidechain_CMR), 
            vector_angle=angle_between_two_vectors(residue_1.vector, residue_2.vector)
        )

    return data_of_all_pairs_of_residues_in_motif

def get_minimum_spanning_tree(all_pairs_of_residues_data: Dict[Tuple[str,str], Residue_pair_data]) -> nx.Graph:
    """
    Returns the Minimum Spanning Tree (MST) of a motif by creating a dense graph of all the residue pairs in the motif and then applying Kruskal's 
    algorithm using the C alpha distance between the residues. If the motif has 3 residues of less then we don't apply minimum spanning tree and 
    simply return the dense graph, otherwise the graph would not be constrained enough and could result in many false positives. The label
    of the nodes in the graph are full residue IDs, that is '<residue_ID><residue_name>' (ex: 'A12G'), and the edges contain residue pair data.
    """
    graph = nx.Graph()
    for (res1_full_ID, res2_full_ID), residue_pair_data in all_pairs_of_residues_data.items():
        graph.add_edge(
            res1_full_ID, res2_full_ID, 
            weight=residue_pair_data.C_alpha_distance, 
            residue_pair_data=residue_pair_data
        )

    if len(graph) <= 3:
        return graph
    
    minimum_spanning_tree = nx.minimum_spanning_tree(G=graph, algorithm='kruskal')

    return minimum_spanning_tree

def get_nodes_original_residue_map(nodes: List[str]) -> Dict[str,str]:
    # Ex: 'A12P' -> 'A12':'P'
    return {node[:-1]:node[-1] for node in nodes}

def get_nodes_position_specific_exchanges(nodes: List[str], residue_type_policy: Union[str, Dict[str,List[str]]], motif_residues_data: Dict[str, Residue]) -> Dict[str,List[str]]:
    """
    """
    nodes_position_specific_exchanges: Dict[str,List[str]] = {}
    if type(residue_type_policy) is dict:
        for node in nodes:
            node_ID, reference_residue = node[:-1], node[-1]
            if node_ID in residue_type_policy: # The user has specified alternative residues for this residue ID
                alternative_residues: Set[str] = set(residue_type_policy[node_ID])
                alternative_residues.discard(reference_residue) # Make sure the original residue is not included
                nodes_position_specific_exchanges[node] = [get_full_residue_ID(node_ID, resname) for resname in alternative_residues]

            else: # The user hasn't specified any mutations for this node
                nodes_position_specific_exchanges[node] = []

    elif residue_type_policy == 'relaxed':
        for node in nodes:
            node_ID, reference_residue = node[:-1], node[-1]
            nodes_position_specific_exchanges[node] = [
                get_full_residue_ID(node_ID, resname) 
                for resname in AMINO_ACID_RELAXED_GROUPS_MAP[reference_residue]
                if resname != reference_residue # Make sure the original residue is not included
            ]

    elif residue_type_policy == 'fully_relaxed':
        for node in nodes:
            node_ID, reference_residue = node[:-1], node[-1]
            nodes_position_specific_exchanges[node] = [
                get_full_residue_ID(node_ID, resname) 
                for resname in AMINO_ACID_ALPHABET
                if resname != reference_residue
            ]

    return nodes_position_specific_exchanges

def get_mutations_mapping(nodes_to_mutate: Tuple[str,...], nodes_original_residue_map: Dict[str,str]) -> Dict[str,str]:
    """
    """
    mutations_mapping = {}
    for node in nodes_to_mutate:
        node_ID = node[:-1] # Ex: 'A12'
        node_original_residue = nodes_original_residue_map[node_ID]
        mutations_mapping[get_full_residue_ID(node_ID, node_original_residue)] = node
    
    return mutations_mapping

def get_all_motif_MSTs_generator(motif_MST: nx.Graph, max_n_mutated_residues: int, residue_type_policy: Union[str, Dict[str,List[str]]], motif_residues_data: Dict[str, Residue]) -> Iterator[nx.Graph]:
    """
    ...
    """
    # No mater what, we will always have to solve the initial motif itself, and we can end the generator here if no variants 
    # of the motif need to be generated, ie if we are in 'strict' mode.
    yield motif_MST
    if residue_type_policy == 'strict':
        return

    nodes: List[str] = list(motif_MST.nodes)
    nodes_original_residue_map = get_nodes_original_residue_map(nodes)
    nodes_position_specific_exchanges: Dict[str,List[str]] = get_nodes_position_specific_exchanges(nodes, residue_type_policy, motif_residues_data)
    
    # We only need to check nodes that have at least 1 possible mutated residue.
    nodes_with_position_specific_exchanges = [node for node, exchanges in nodes_position_specific_exchanges.items() if len(exchanges) >= 1]
    if max_n_mutated_residues > len(nodes_with_position_specific_exchanges):
        raise ValueError(f'max_n_mutated_residues was set to {max_n_mutated_residues}, but only {len(nodes_with_position_specific_exchanges)} residues are mutable with the given residue_type_policy.')
    

    for n_mutated_residues in range(1, max_n_mutated_residues+1):
        for node_combination in itertools.combinations(nodes_with_position_specific_exchanges, n_mutated_residues): 
            all_mutations_to_perform: List[List[str]] = [nodes_position_specific_exchanges[node] for node in node_combination]
            for nodes_to_mutate in itertools.product(*all_mutations_to_perform, repeat=1):
                mutated_graph: nx.Graph = nx.relabel_nodes(
                    motif_MST, 
                    mapping=get_mutations_mapping(nodes_to_mutate, nodes_original_residue_map), 
                    copy=True
                )
                
                yield mutated_graph
    
def get_all_pairs_of_residues_in_motif_MST(motif_MST: nx.Graph) -> Dict[Tuple[str,str], Residue_pair_data]:
    """
    """
    all_pairs_of_residues_in_motif_MST: Dict[Tuple[str,str], Residue_pair_data] = {}

    MST_edges: List[Tuple[str, str]] = list(motif_MST.edges) # Ex: [('A1G', 'A3K'), ('A3K', 'A8N'), ...]
    for pair_of_full_residue_IDs in MST_edges:
        pair_of_full_residue_IDs = get_sorted_2_tuple(pair_of_full_residue_IDs) # Ordering is needed to make sure we only calculate each pair once, i.e avoid calculating the results of (A,B) and (B,A), as they are identical.
        
        all_pairs_of_residues_in_motif_MST[pair_of_full_residue_IDs] = motif_MST.edges[pair_of_full_residue_IDs]['residue_pair_data']

    return all_pairs_of_residues_in_motif_MST

def clip_angle_values(min_vector_angle: float, max_vector_angle: float) -> Tuple[float, float]:
    """
    ...
    """
    # All angle values in the index are in the range [0,180], and therefore angles > or < to this range have to be mapped to their corresponding
    # small angle equivalent (ex: 185° -> 175°, 270° -> 90°, ...), but when doing so we can observe that we are guaranteed they will 
    # always be > than the other paired angle in the case of angles > 180°, and vice versa for angles < 0°. Thus we can simply clip the min_vector_angle
    # and max_vector_angle to 0 and 180 respectively.
    # Angle < 180° example:   2° +/- 20° -> [-18,  22] -> We map -18° to  18° -> [  0,  18] U [  0,  22] == [  0,  22]
    # Angle > 180° example: 175° +/- 20° -> [155, 195] -> We map 195° to 165° -> [165, 180] U [155, 180] == [155, 180]
    min_vector_angle = 0.0 if min_vector_angle < 0 else min_vector_angle
    max_vector_angle = 180.0 if max_vector_angle > 180 else max_vector_angle

    return min_vector_angle, max_vector_angle

def get_bin_data(min_geometric_descriptor_value: float, max_geometric_descriptor_value: float, bin_size: float) -> Tuple[List[int], int, int]:
    """
    ...
    """
    min_bin = get_bin_number(min_geometric_descriptor_value, bin_size)
    max_bin = get_bin_number(max_geometric_descriptor_value, bin_size)

    return [bin for bin in range(min_bin, max_bin+1)], min_bin, max_bin

def trim_dataframe(
        df: pd.DataFrame, bin_value: int, min_bin_value: int, max_bin_value: int, 
        min_geometric_descriptor_value: float, max_geometric_descriptor_value: float, column_name: str
    ) -> pd.DataFrame:
    """
    ...
    """
    if bin_value == min_bin_value:
        df = df[df[column_name] >= min_geometric_descriptor_value]
    elif bin_value == max_bin_value:
        df = df[df[column_name] <= max_geometric_descriptor_value]

    # If the bin value is neither the min or max bin value, then all the rows are correct for this geometric descriptor and we therefore don't have to do anything

    return df

def dataframes_within_tolerance_ranges_generator(
        pair_of_residues: str, compression: str, index_folder_path: Path, min_C_alpha_distance: float, max_C_alpha_distance: float, 
        min_sidechain_CMR_distance: float, max_sidechain_CMR_distance:float, min_vector_angle: float, max_vector_angle: float
    ) -> Iterator[pd.DataFrame]:
    """
    ...
    """
    C_alpha_distance_bins_list, min_C_alpha_distance_bin, max_C_alpha_distance_bin = get_bin_data(min_C_alpha_distance, max_C_alpha_distance, bin_size=INDEX_DISTANCE_BIN_SIZE)
    sidechain_CMR_distance_bins_list, min_sidechain_CMR_distance_bin, max_sidechain_CMR_distance_bin  = get_bin_data(min_sidechain_CMR_distance, max_sidechain_CMR_distance, bin_size=INDEX_DISTANCE_BIN_SIZE)
    vector_angle_bins_list, min_vector_angle_bin, max_vector_angle_bin = get_bin_data(min_vector_angle, max_vector_angle, bin_size=INDEX_ANGLE_BIN_SIZE)

    for bin_combination in itertools.product(C_alpha_distance_bins_list, sidechain_CMR_distance_bins_list, vector_angle_bins_list):
        C_alpha_distance_bin, sidechain_CMR_distance_bin, vector_angle_bin = bin_combination

        index_file = index_folder_path / 'index' / f'{pair_of_residues}_{C_alpha_distance_bin}_{sidechain_CMR_distance_bin}_{vector_angle_bin}.{compression}' # e.g /home/user_name/database_folder/pyScoMotif_index/index/AG_4_4_5.bz2
        if not index_file.exists():
            yield pd.DataFrame() # Empty dataframe
            continue
        
        # In order to only get the residue pairs that are exactly within the tolerance ranges specified by the user (e.g: +/- 2.0 Å for distance descriptors), 
        # we have to be careful with dataframes that have a bin number that is either the min or max bin number of a given geometric descriptor. 
        # For example, if the min_C_alpha_distance = 2.3 and we load the files with the corresponding bin number (i.e 2.3 // 1.0 = 2 = bin number), those dataframes 
        # will contain rows with 2 < C_alpha_distance < 2.99, so we must remove the rows with C_alpha_distance < 2.3 (i.e trim the dataframe). 
        # We have to do this process for each geometric descriptor to get correct dataframes.
        df: pd.DataFrame = read_compressed_and_pickled_file(index_file)

        df = trim_dataframe(df, C_alpha_distance_bin, min_C_alpha_distance_bin, max_C_alpha_distance_bin, min_C_alpha_distance, max_C_alpha_distance, column_name='C_alpha_distance')
        df = trim_dataframe(df, sidechain_CMR_distance_bin, min_sidechain_CMR_distance_bin, max_sidechain_CMR_distance_bin, min_sidechain_CMR_distance, max_sidechain_CMR_distance, column_name='sidechain_CMR_distance')
        df = trim_dataframe(df, vector_angle_bin, min_vector_angle_bin, max_vector_angle_bin, min_vector_angle, max_vector_angle, column_name='vector_angle')

        yield df

def get_PDBs_that_contain_the_residue_pair(
        pair_of_full_residue_IDs: Tuple[str,str], residue_pair_data: Residue_pair_data, distance_delta_thr: float, angle_delta_thr: float, 
        index_folder_path: Path, compression: str
    ) -> Dict[str, List[Tuple[str,str]]]:
    """
    ...
    """
    res1_full_ID, res2_full_ID = pair_of_full_residue_IDs
    pair_of_residues = sort_and_join_2_strings(res1_full_ID[-1], res2_full_ID[-1]) # The index only contains data of lexycographically sorted reside pairs (e.g: GA index files don't exist, only AG files do)

    # To finds all the PDBs that contain the given residue pair in the correct geometric arragement, we read all the dataframes of the residue pair that 
    # are within the tolerance thresholds (specified by the user) and concatenate them into a single full dataframe
    min_C_alpha_distance, max_C_alpha_distance = residue_pair_data.C_alpha_distance - distance_delta_thr, residue_pair_data.C_alpha_distance + distance_delta_thr
    min_sidechain_CMR_distance, max_sidechain_CMR_distance = residue_pair_data.sidechain_CMR_distance - distance_delta_thr, residue_pair_data.sidechain_CMR_distance + distance_delta_thr
    min_vector_angle, max_vector_angle = residue_pair_data.vector_angle - angle_delta_thr, residue_pair_data.vector_angle + angle_delta_thr
    min_vector_angle, max_vector_angle = clip_angle_values(min_vector_angle, max_vector_angle)
    
    full_residue_pair_df = pd.concat(
        objs=dataframes_within_tolerance_ranges_generator(pair_of_residues, compression, index_folder_path, min_C_alpha_distance, max_C_alpha_distance, min_sidechain_CMR_distance, max_sidechain_CMR_distance, min_vector_angle, max_vector_angle), 
        axis=0, 
        copy=False
    )
    if full_residue_pair_df.empty:
        return {}

    PDBs_that_contain_the_residue_pair: Dict[str, List[Tuple[str,str]]] = defaultdict(list)
    res1_resname, res2_resname = pair_of_residues[0], pair_of_residues[1]
    res1_ID: str; res2_ID: str; PDB_ID: str
    for res1_ID, res2_ID, PDB_ID in zip(full_residue_pair_df.residue_1.values, full_residue_pair_df.residue_2.values, full_residue_pair_df.PDB_ID.values):
        res1_full_ID = res1_ID + res1_resname # Ex: 'A1G', that is residue A1 which is a Glycine
        res2_full_ID = res2_ID + res2_resname
        pair_of_full_residue_IDs = get_sorted_2_tuple((res1_full_ID, res2_full_ID)) # Sorting is needed to be able to find PDBs with the residue pair in the correct geometric arrangement later
        
        PDBs_that_contain_the_residue_pair[PDB_ID].append(pair_of_full_residue_IDs)
        
    return PDBs_that_contain_the_residue_pair

def update_map_of_PDBs_with_all_residue_pairs(
        map_of_PDBs_with_all_residue_pairs: Dict[str, List[Tuple[str,str]]], new_residue_pair_data: Dict[str, List[Tuple[str,str]]]
    ) -> None:
    """
    ...
    """
    updated_set_of_PDBs_with_all_residue_pairs = map_of_PDBs_with_all_residue_pairs.keys() & new_residue_pair_data.keys() # & = set intersection 
    
    for PDB_ID in list(map_of_PDBs_with_all_residue_pairs.keys()):
        if PDB_ID not in updated_set_of_PDBs_with_all_residue_pairs:
            del map_of_PDBs_with_all_residue_pairs[PDB_ID] # We remove PDBs that no longer have all the residue pairs
        else:
            map_of_PDBs_with_all_residue_pairs[PDB_ID].extend(new_residue_pair_data[PDB_ID]) # We update the set of residue pairs of the PDBs that have all the residue pairs

    return

def find_PDBs_with_all_residue_pairs(
        motif_MST: nx.Graph, index_folder_path: Path, distance_delta_thr: float, angle_delta_thr: float, compression: str, 
        concurrent_executor: ProcessPoolExecutor
    ) -> Dict[str, List[Tuple[str,str]]]:
    """
    """
    # The algorithm determines which PDBs contain all the residue pairs of a given motif MST by simply iterating over each residue pair 
    # and calculating the intersection between the current map of PDBs with all residue pairs and the PDBs that contain
    # the given residue pair that is being checked (i.e iterative intersection).
    all_pairs_of_residues_to_check = get_all_pairs_of_residues_in_motif_MST(motif_MST)
    
    submitted_futures: List[Future[Dict[str, List[Tuple[str,str]]]]] = [
        concurrent_executor.submit(get_PDBs_that_contain_the_residue_pair, pair_of_full_residue_IDs, residue_pair_data, distance_delta_thr, angle_delta_thr, index_folder_path, compression)
        for pair_of_full_residue_IDs, residue_pair_data in all_pairs_of_residues_to_check.items()
    ]

    map_of_PDBs_with_all_residue_pairs: Dict[str, List[Tuple[str,str]]]
    map_of_PDBs_with_all_residue_pairs_initialized_flag: bool = False
    for future in as_completed(submitted_futures):
        if future.exception():
            raise future.exception() # type: ignore

        PDBs_that_contain_the_residue_pair = future.result()

        # Iterative intersection code. map_of_PDBs_with_all_residue_pairs must be initialized with the PDBs of a 
        # residue pair in order to be able to do the iterative intersection, otherwise we would always get an empty set 
        # given it is initially empty.
        if map_of_PDBs_with_all_residue_pairs_initialized_flag == False:
            map_of_PDBs_with_all_residue_pairs = defaultdict(list, PDBs_that_contain_the_residue_pair)
            map_of_PDBs_with_all_residue_pairs_initialized_flag = True

        else:
            update_map_of_PDBs_with_all_residue_pairs(
                map_of_PDBs_with_all_residue_pairs, 
                PDBs_that_contain_the_residue_pair
            )
        
    return map_of_PDBs_with_all_residue_pairs

def add_resname_as_node_attribute(graph: nx.Graph) -> None:
    """
    """
    nx.set_node_attributes(graph, {node:{'resname':node[-1]} for node in graph.nodes}) # Each node is a residue_full_ID so [-1] returns the residue (ex: 'G', for Glycine)
    return

def run_subgraph_monomorphism(motif_MST: nx.Graph, pairs_of_residues: List[Tuple[str,str]]) -> Union[None, List[nx.Graph]]:
    """
    The returned list can be empty.
    """
    candidate_PDB_graph = nx.Graph(pairs_of_residues)
    add_resname_as_node_attribute(candidate_PDB_graph)

    graph_matcher = nx.algorithms.isomorphism.GraphMatcher(
        G1 = candidate_PDB_graph, G2 = motif_MST, 
        node_match = lambda node1,node2: node1['resname'] == node2['resname'] # Match based on residue name
    )

    monomorphism_checked_motifs: List[nx.Graph] = [] # Each PDB can have multiple motifs (i.e: homodimers) -> List[nx.Graph]
    residue_mapping_dict: Dict[str,str]
    for residue_mapping_dict in graph_matcher.subgraph_monomorphisms_iter():
        residues_of_the_similar_motif = list(residue_mapping_dict.keys())
        similar_motif_graph: nx.Graph = candidate_PDB_graph.subgraph(residues_of_the_similar_motif).copy()
        
        setattr(similar_motif_graph, 'residue_mapping_dict', residue_mapping_dict) # Used for RMSD calculation

        monomorphism_checked_motifs.append(similar_motif_graph)

    return monomorphism_checked_motifs if monomorphism_checked_motifs else None

def filter_out_PDBs_with_unconnected_residue_pairs(
        PDBs_with_all_residue_pairs: Dict[str, List[Tuple[str,str]]], motif_MST: nx.Graph, concurrent_executor: ProcessPoolExecutor, 
    ) -> Dict[str, List[nx.Graph]]:
    """
    """
    # To know if a PDB contains a set of connected residue pairs forming a similar motif, we use subgraph monomorphism to perform a matching between a reference 
    # motif's graph and the graph generated from all the selected residue pairs of a candidate PDB structure. The monomorphism is done based on the residue types,
    # which have to be identical between the two graphs. Note that we use subgraph monomorphism instead of simple graph monomorphism because PDBs might have more 
    # than 1 similar motif, for example a motif might be present twice in a homodimer, so we need to check all possible subgraphs in the graph generated from the selected residue pairs.
    # Also note that isomorphism cannot be used as it imposes an exact match of both the nodes and edges between all the nodes, which results in certain False Negatives
    # given the edges simply correspond to the presence of a pair of residues, they are not chemical bonds. For example, sub-graph isomorphism fails to find a match for residues 31-38 in
    # PDB AF-G5EB01-F1-model_v4 despite the PDB being in the index. Indeed, when using a distance threshold > 1.5 and an angle threshold > 10°, this leads to 
    # the occurence of an additional AE edge between residue A32A and A35E, and that edge causes isomorphism to fail, unlike monomorphism which correctly finds the PDB.
    add_resname_as_node_attribute(motif_MST) # Needed for subgraph monomorphism, see run_subgraph_monomorphism().

    submited_futures: Dict[Future[Union[None, List[nx.Graph]]], str] = {
        concurrent_executor.submit(run_subgraph_monomorphism, motif_MST, pairs_of_residues):PDB_ID
        for PDB_ID, pairs_of_residues in PDBs_with_all_residue_pairs.items()
    }

    filtered_PDBs_with_all_residue_pairs: Dict[str, List[nx.Graph]] =  {}
    for future in as_completed(submited_futures):
        if future.exception():
            raise future.exception() # type: ignore

        monomorphism_checked_motifs = future.result()
        if not monomorphism_checked_motifs: # These are the false positive PDBs, i.e PDBs that have all the residue pairs but where the monomorphism check doesn't find any similar motif as a result of unconnected pairs of residues 
            continue

        PDB_ID = submited_futures[future]
        filtered_PDBs_with_all_residue_pairs[PDB_ID] = monomorphism_checked_motifs

    return filtered_PDBs_with_all_residue_pairs

def residue_pair_level_parallelisation(
        reference_motif_MST: nx.Graph, max_n_mutated_residues: int, residue_type_policy: Union[str, Dict[str,List[str]]],
        reference_motif_residues_data: Dict[str, Residue], index_folder_path: Path, distance_delta_thr: float, 
        angle_delta_thr: float, compression: str, concurrent_executor: ProcessPoolExecutor, 
        motif_MST_filtered_PDBs_map: Dict[nx.Graph, Dict[str, List[nx.Graph]]]
    ) -> None:
    """
    ...
    """
    for motif_MST in get_all_motif_MSTs_generator(reference_motif_MST, max_n_mutated_residues, residue_type_policy, reference_motif_residues_data):
        PDBs_with_all_residue_pairs = find_PDBs_with_all_residue_pairs(motif_MST, index_folder_path, distance_delta_thr, angle_delta_thr, compression, concurrent_executor)
        filtered_PDBs_with_all_residue_pairs = filter_out_PDBs_with_unconnected_residue_pairs(PDBs_with_all_residue_pairs, motif_MST, concurrent_executor)
        
        if filtered_PDBs_with_all_residue_pairs:
            motif_MST_filtered_PDBs_map[motif_MST] = filtered_PDBs_with_all_residue_pairs
    
    return

def get_reference_motif_residue_pairs_data(
        reference_motif_MST: nx.Graph, index_folder_path: Path, distance_delta_thr: float, angle_delta_thr: float, 
        compression: str, concurrent_executor: ProcessPoolExecutor
    ) -> Tuple[List[bytes], Dict[Tuple[str,str], int]]:
    """
    ...
    """
    # First load all the residue pair data of the reference_motif_MST
    all_pairs_of_residues_to_check = get_all_pairs_of_residues_in_motif_MST(reference_motif_MST)
    
    submitted_futures: Dict[Future[Dict[str, List[Tuple[str,str]]]], Tuple[str,str]] = {
        concurrent_executor.submit(get_PDBs_that_contain_the_residue_pair, pair_of_full_residue_IDs, residue_pair_data, distance_delta_thr, angle_delta_thr, index_folder_path, compression):pair_of_full_residue_IDs
        for pair_of_full_residue_IDs, residue_pair_data in all_pairs_of_residues_to_check.items()
    }

    # The dictionary of each residue pair has to be turned into bytes in order to be shareable across multiple processes, raw 
    # dictionaries are not accepted by ShareableList. Also, given we are sharing a list, we need to keep a mapping between each 
    # residue pair data and it's index position in the list (e.g: the data of residue pair ('A1', 'A2') is at index position [1]).
    reference_motif_residue_pairs_data: List[bytes] = []
    shared_memory_residue_pair_mapping: Dict[Tuple[str,str], int] = {}
    for i, future in enumerate(as_completed(submitted_futures)):
        if future.exception():
            raise future.exception() # type: ignore

        PDBs_that_contain_the_residue_pair = future.result()
        pair_of_full_residue_IDs = submitted_futures[future]

        reference_motif_residue_pairs_data.append(pickle.dumps(PDBs_that_contain_the_residue_pair)) # Byte encode with pickle
        shared_memory_residue_pair_mapping[pair_of_full_residue_IDs] = i

    return reference_motif_residue_pairs_data, shared_memory_residue_pair_mapping

def load_data_from_shared_memory(
        pair_of_full_residue_IDs: Tuple[str,str], shared_memory_residue_pair_mapping:Dict[Tuple[str,str],int], 
        shared_memory_list: ShareableList[bytes]
    ) -> Dict[str, List[Tuple[str, str]]]:
    """
    ...
    """
    list_index_position = shared_memory_residue_pair_mapping[pair_of_full_residue_IDs]
    bytes_data = shared_memory_list[list_index_position]
    PDBs_that_contain_the_residue_pair: Dict[str, List[Tuple[str, str]]] = pickle.loads(bytes_data)
    return PDBs_that_contain_the_residue_pair

def solve_motif_MST(
        motif_MST: nx.Graph, index_folder_path: Path, distance_delta_thr: float, angle_delta_thr: float, compression: str, 
        shared_memory_list_name: str, shared_memory_residue_pair_mapping: Dict[Tuple[str,str], int] 
    ) -> Dict[str, List[nx.Graph]]:
    """
    ...
    """
    # Non-parallel version of find_PDBs_with_all_residue_pairs
    map_of_PDBs_with_all_residue_pairs: Dict[str, List[Tuple[str,str]]]
    map_of_PDBs_with_all_residue_pairs_initialized_flag: bool = False
    shared_memory_list: ShareableList[bytes] = ShareableList(name=shared_memory_list_name) # 'Connect' to the shared memory block
    for pair_of_full_residue_IDs, residue_pair_data in get_all_pairs_of_residues_in_motif_MST(motif_MST).items():
        if pair_of_full_residue_IDs in shared_memory_residue_pair_mapping:
            PDBs_that_contain_the_residue_pair = load_data_from_shared_memory(pair_of_full_residue_IDs, shared_memory_residue_pair_mapping, shared_memory_list)
        else:
            PDBs_that_contain_the_residue_pair = get_PDBs_that_contain_the_residue_pair(pair_of_full_residue_IDs, residue_pair_data, distance_delta_thr, angle_delta_thr, index_folder_path, compression)
        

        if map_of_PDBs_with_all_residue_pairs_initialized_flag == False:
            map_of_PDBs_with_all_residue_pairs = defaultdict(list, PDBs_that_contain_the_residue_pair)
            map_of_PDBs_with_all_residue_pairs_initialized_flag = True
        else:
            update_map_of_PDBs_with_all_residue_pairs(
                map_of_PDBs_with_all_residue_pairs, 
                PDBs_that_contain_the_residue_pair
            )
    # Clean up
    shared_memory_list.shm.close() # 'Disconnect'
    del shared_memory_residue_pair_mapping, shared_memory_list

    # Non-parallel version of filter_out_PDBs_with_unconnected_residue_pairs
    add_resname_as_node_attribute(motif_MST) # Needed for subgraph monomorphism, see run_subgraph_monomorphism().
    filtered_PDBs_with_all_residue_pairs: Dict[str, List[nx.Graph]] =  {}
    for PDB_ID, pairs_of_residues in map_of_PDBs_with_all_residue_pairs.items():
        monomorphism_checked_motifs = run_subgraph_monomorphism(motif_MST, pairs_of_residues)

        if not monomorphism_checked_motifs: # These are the false positive PDBs, i.e PDBs that have all the residue pairs but where the monomorphism check doesn't find any similar motif as a result of unconnected pairs of residues 
            continue

        filtered_PDBs_with_all_residue_pairs[PDB_ID] = monomorphism_checked_motifs

    return filtered_PDBs_with_all_residue_pairs
    
def motif_level_parallelisation(        
        reference_motif_MST: nx.Graph, max_n_mutated_residues: int, residue_type_policy: Union[str, Dict[str,List[str]]],
        reference_motif_residues_data: Dict[str, Residue], index_folder_path: Path, distance_delta_thr: float, 
        angle_delta_thr: float, compression: str, concurrent_executor: ProcessPoolExecutor, 
        motif_MST_filtered_PDBs_map: Dict[nx.Graph, Dict[str, List[nx.Graph]]]) -> None:
    """
    ...
    """
    # When dealing with many mutated versions of a reference motif, the data of the same residue pairs is often loaded over
    # and over again, in particular the residue pair data of the reference motif. Given IO is one of the main speed 
    # bottlenecks of the search procedure, loading this data once and sharing it between the parallel processes can speed 
    # up the search. For that we use ShareableList, which enables different processes to access the same data directly from RAM.
    reference_motif_residue_pairs_data, shared_memory_residue_pair_mapping = get_reference_motif_residue_pairs_data(
        reference_motif_MST,
        index_folder_path,
        distance_delta_thr, angle_delta_thr,
        compression,
        concurrent_executor
    )
    shared_memory_list = ShareableList(reference_motif_residue_pairs_data)
    shared_memory_list_name: str = shared_memory_list.shm.name
    
    # Motif level parallelisation code
    try:
        submitted_futures = {
            concurrent_executor.submit(solve_motif_MST, motif_MST, index_folder_path, distance_delta_thr, angle_delta_thr, compression, shared_memory_list_name, shared_memory_residue_pair_mapping):motif_MST
            for motif_MST in get_all_motif_MSTs_generator(reference_motif_MST, max_n_mutated_residues, residue_type_policy, reference_motif_residues_data)
        }

        for future in as_completed(submitted_futures):
            if future.exception():
                raise future.exception() # type: ignore

            filtered_PDBs_with_all_residue_pairs = future.result()
            if filtered_PDBs_with_all_residue_pairs:
                motif_MST = submitted_futures[future]
                motif_MST_filtered_PDBs_map[motif_MST] = filtered_PDBs_with_all_residue_pairs
    
    except Exception as exception:
        raise exception

    finally:
        # Clean up the shared memory
        shared_memory_list.shm.close()
        shared_memory_list.shm.unlink()

    return

def get_PDBs_with_similar_motifs(
        reference_motif_MST: nx.Graph, reference_motif_residues_data: Dict[str, Residue], index_folder_path: Path, max_n_mutated_residues: int, 
        residue_type_policy: Union[str, Dict[str,List[str]]], distance_delta_thr: float, angle_delta_thr: float, compression: str, n_cores: int
    ) -> Dict[nx.Graph, Dict[str, List[nx.Graph]]]:
    """
    ...
    """
    motif_MST_filtered_PDBs_map: Dict[nx.Graph, Dict[str, List[nx.Graph]]] = {}
    n_motifs_to_solve = sum(1 for _ in get_all_motif_MSTs_generator(reference_motif_MST, max_n_mutated_residues, residue_type_policy, reference_motif_residues_data))
    with ProcessPoolExecutor(max_workers=n_cores) as concurrent_executor:
        if n_motifs_to_solve < n_cores:
            residue_pair_level_parallelisation(
                reference_motif_MST, 
                max_n_mutated_residues, residue_type_policy, 
                reference_motif_residues_data, 
                index_folder_path, 
                distance_delta_thr, angle_delta_thr, 
                compression, 
                concurrent_executor,
                motif_MST_filtered_PDBs_map
            )

        # When having to check a large number of motifs as a result of mutated residues provided by the user, 
        # motif level parallelisation is ~35% faster than residue-pair level parallelisation.
        else:
            motif_level_parallelisation(
                reference_motif_MST, 
                max_n_mutated_residues, residue_type_policy, 
                reference_motif_residues_data, 
                index_folder_path, 
                distance_delta_thr, angle_delta_thr, 
                compression, 
                concurrent_executor,
                motif_MST_filtered_PDBs_map
            )
            
    return motif_MST_filtered_PDBs_map

def search_index_for_PDBs_with_similar_motifs(
        index_folder_path: Path, PDB_file: Path, motif: Tuple[str,...], residue_type_policy: Union[str, Dict[str,List[str]]], max_n_mutated_residues: int, 
        distance_delta_thr: float, angle_delta_thr: float, n_cores: int
    ) -> Tuple[nx.Graph, Dict[nx.Graph, Dict[str, List[nx.Graph]]]]:
    """
    ...
    """
    if not index_folder_exists(index_folder_path):
        raise ValueError("Could not find all the index folders and files. Did you previously create the index with the 'create-index' command ? Does the index path point towards a pyScoMotif_index folder ?")
    
    reference_motif_residues_data = extract_motif_residues_from_PDB_file(PDB_file, motif)
    data_of_all_pairs_of_residues_in_motif = get_data_of_all_pairs_of_residues_in_motif(reference_motif_residues_data)

    # It would be inefficient to search the index for every single pair of residues in the motif in order to find PDBs that have a similar 
    # motif. Instead we can determine the Minimum Spanning Tree (MST) of the motif to limit the search to a subset of pairs that covers all the residues.
    reference_motif_MST = get_minimum_spanning_tree(data_of_all_pairs_of_residues_in_motif)

    compression = detect_the_compression_algorithm_used_in_the_index(index_folder_path)
    
    PDBs_with_similar_motifs = get_PDBs_with_similar_motifs(
        reference_motif_MST, 
        reference_motif_residues_data, 
        index_folder_path,
        max_n_mutated_residues, 
        residue_type_policy,
        distance_delta_thr, 
        angle_delta_thr,
        compression,
        n_cores
    )

    return reference_motif_MST, PDBs_with_similar_motifs
