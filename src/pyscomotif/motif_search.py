import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple, Union

import networkx as nx
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from pyscomotif.constants import (AMINO_ACID_ALPHABET,
                                  AMINO_ACID_RELAXED_GROUPS_MAP,
                                  INDEX_ANGLE_BIN_SIZE,
                                  INDEX_DISTANCE_BIN_SIZE)
from pyscomotif.data_containers import Residue, Residue_pair_data
from pyscomotif.index_folders_and_files import index_folder_exists
from pyscomotif.residue_data_dicts import extract_residue_data
from pyscomotif.utils import (
    angle_between_two_vectors,
    detect_the_compression_algorithm_used_in_the_index, flatten_iterable,
    get_bin_number, get_sorted_2_tuple, pairwise_euclidean_distance,
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
    Returns all the residue pairs in motif_MST in an ordered manner such that, for all i, residue pairs between 0 and i form a connected graph. The order follows 
    that of a Depth First Search (dfs). If the residue pairs were randomly ordered, we wouldn't be able to do the connectivity check 
    in update_map_of_PDBs_with_all_residue_pairs, because the logic requires that new residue pairs are connected to already visited nodes.
    """
    MST_edges: List[Tuple[str, str]] = list(nx.edge_dfs(motif_MST)) # Ex: [('A1G', 'A3K'), ('A3K', 'A8N'), ...]

    all_pairs_of_residues_in_motif_MST: Dict[Tuple[str,str], Residue_pair_data] = {}
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
        res1_ID, res2_ID = str(res1_ID), str(res2_ID) # Some PDBs such as 3PGA name their chains using numbers, which results in some dataframes where the residue_1 and residue_2 columns are not strings but ints.

        res1_full_ID = res1_ID + res1_resname # Ex: 'A1G', that is residue A1 which is a Glycine
        res2_full_ID = res2_ID + res2_resname
        pair_of_full_residue_IDs = get_sorted_2_tuple((res1_full_ID, res2_full_ID)) # Sorting is needed to be able to find PDBs with the residue pair in the correct geometric arrangement later
        
        PDBs_that_contain_the_residue_pair[PDB_ID].append(pair_of_full_residue_IDs)
        
    return PDBs_that_contain_the_residue_pair

def residue_pair_is_connected_to_visited_nodes(residue_pair: Tuple[str, str], visited_nodes: Set[str]) -> bool:
    return residue_pair[0] in visited_nodes or residue_pair[1] in visited_nodes

def update_map_of_PDBs_with_all_residue_pairs(
        map_of_PDBs_with_all_residue_pairs: Dict[str, List[Tuple[str,str]]], new_residue_pair_data: Dict[str, List[Tuple[str,str]]], visited_nodes_map: Dict[str, Set[str]]
    ) -> None:
    """
    ...
    """
    updated_set_of_PDBs_with_all_residue_pairs = map_of_PDBs_with_all_residue_pairs.keys() & new_residue_pair_data.keys() # & = set intersection
    
    for PDB_ID in list(map_of_PDBs_with_all_residue_pairs.keys()):
        # Drop PDBs that no longer have all the residue pairs
        if PDB_ID not in updated_set_of_PDBs_with_all_residue_pairs:
            del map_of_PDBs_with_all_residue_pairs[PDB_ID]
            del visited_nodes_map[PDB_ID]
            continue
        
        # Connectivity check. New residue pairs that are not connected to already visited nodes can be dropped, and if all are droped then we can also drop the PDB
        connectivity_checked_residue_pairs: List[Tuple[str,str]] = [
            residue_pair
            for residue_pair in new_residue_pair_data[PDB_ID]
            if residue_pair_is_connected_to_visited_nodes(residue_pair, visited_nodes_map[PDB_ID])
        ]
        if not connectivity_checked_residue_pairs:
            del map_of_PDBs_with_all_residue_pairs[PDB_ID]
            del visited_nodes_map[PDB_ID]
            continue

        map_of_PDBs_with_all_residue_pairs[PDB_ID].extend(connectivity_checked_residue_pairs) # Update the list of residue pairs of the PDB
        visited_nodes_map[PDB_ID].update(set(flatten_iterable(connectivity_checked_residue_pairs)))

    return

def find_PDBs_with_all_residue_pairs(
        motif_MST: nx.Graph, index_folder_path: Path, distance_delta_thr: float, angle_delta_thr: float, compression: str, parallel: Parallel
    ) -> Dict[str, List[Tuple[str,str]]]:
    """
    """
    # The algorithm determines which PDBs contain all the residue pairs of a given motif MST by iterating over each residue pair and calculating the 
    # intersection between the current map of PDBs with all residue pairs and the PDBs that contain the given residue pair that is being 
    # checked (i.e iterative intersection). A connectivity check between the already determined residue pairs and the new ones is also performed to remove PDBs 
    # that have the reside pairs but are not connected and therefore don't form a motif (see update_map_of_PDBs_with_all_residue_pairs).
    all_pairs_of_residues_to_check = get_all_pairs_of_residues_in_motif_MST(motif_MST)
    delayed_func: Callable[[Tuple[str, str], Residue_pair_data, float, float, Path, str], Dict[str, List[Tuple[str, str]]]] = delayed(get_PDBs_that_contain_the_residue_pair)
    results_generator: Iterator[Dict[str, List[Tuple[str,str]]]] = parallel(
        delayed_func(pair_of_full_residue_IDs, residue_pair_data, distance_delta_thr, angle_delta_thr, index_folder_path, compression) 
        for pair_of_full_residue_IDs, residue_pair_data in all_pairs_of_residues_to_check.items()
    )
    
    map_of_PDBs_with_all_residue_pairs: Dict[str, List[Tuple[str,str]]]
    map_of_PDBs_with_all_residue_pairs_initialized_flag: bool = False
    visited_nodes_map: Dict[str, Set[str]] # Used for connectivity check
    for PDBs_that_contain_the_residue_pair in results_generator:
        # Iterative intersection code. map_of_PDBs_with_all_residue_pairs must be initialized with the PDBs of a 
        # residue pair in order to be able to do the iterative intersection, otherwise we would always get an empty set 
        # given it is initially empty.
        if map_of_PDBs_with_all_residue_pairs_initialized_flag == False:
            map_of_PDBs_with_all_residue_pairs = defaultdict(list, PDBs_that_contain_the_residue_pair)
            map_of_PDBs_with_all_residue_pairs_initialized_flag = True
            visited_nodes_map = {
                PDB_ID:set(flatten_iterable(list_of_residue_pairs))
                for PDB_ID, list_of_residue_pairs in map_of_PDBs_with_all_residue_pairs.items()
            }

        else:
            update_map_of_PDBs_with_all_residue_pairs(
                map_of_PDBs_with_all_residue_pairs, 
                PDBs_that_contain_the_residue_pair,
                visited_nodes_map
            )

    return map_of_PDBs_with_all_residue_pairs

def add_resname_as_node_attribute(graph: nx.Graph) -> None:
    """
    """
    nx.set_node_attributes(graph, {node:{'resname':node[-1]} for node in graph.nodes}) # Each node is a residue_full_ID so [-1] returns the residue (ex: 'G', for Glycine)
    return

def run_subgraph_monomorphism(motif_MST: nx.Graph, pairs_of_residues: List[Tuple[str,str]]) -> List[nx.Graph]:
    """
    The returned list can be empty.
    """
    candidate_PDB_graph = nx.Graph(pairs_of_residues)
    add_resname_as_node_attribute(candidate_PDB_graph)

    # NOTE: We don't need to use the edge_matcher functionality of the GraphMatcher because in the previous step we performed a connectivity check,
    # which ensures that each residue pair of the candidate PDB is in the correct geometrical arrangement AND connected to a residue pair. 
    # If the connectivity check had not been performed, candidate PDBs with residue pairs in the correct geometrical arrangement BUT connected in
    # the wrong way would pass the monomorphism check given it does not check edge values, only the residue types. This kind of edge case can for example happen
    # when an epitope has a chain of repeating residue pairs (i.e (S,T),(T,S),(S,T), ...).
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

    return monomorphism_checked_motifs

def filter_out_PDBs_with_unconnected_residue_pairs(
        PDBs_with_all_residue_pairs: Dict[str, List[Tuple[str,str]]], motif_MST: nx.Graph, parallel: Parallel, 
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

    delayed_func: Callable[[nx.Graph, List[Tuple[str, str]]], List[nx.Graph]] = delayed(run_subgraph_monomorphism)
    results_generator: Iterator[List[nx.Graph]] = parallel(
        delayed_func(motif_MST, pairs_of_residues) 
        for PDB_ID, pairs_of_residues in PDBs_with_all_residue_pairs.items()
    )

    filtered_PDBs_with_all_residue_pairs: Dict[str, List[nx.Graph]] =  {}
    for (PDB_ID, pairs_of_residues), monomorphism_checked_motifs in zip(PDBs_with_all_residue_pairs.items(), results_generator, strict=True):
        if not monomorphism_checked_motifs: # These are the false positive PDBs, i.e PDBs that have all the residue pairs but where the monomorphism check doesn't find any similar motif 
            continue

        filtered_PDBs_with_all_residue_pairs[PDB_ID] = monomorphism_checked_motifs

    return filtered_PDBs_with_all_residue_pairs

def solve_motif_MST(
        motif_MST: nx.Graph, index_folder_path: Path, distance_delta_thr: float, angle_delta_thr: float, compression: str, 
    ) -> Dict[str, List[nx.Graph]]:
    """
    ...
    """
    # Non-parallel version of find_PDBs_with_all_residue_pairs
    map_of_PDBs_with_all_residue_pairs: Dict[str, List[Tuple[str,str]]]
    map_of_PDBs_with_all_residue_pairs_initialized_flag: bool = False
    visited_nodes_map: Dict[str, Set[str]] # Used for connectivity check
    for pair_of_full_residue_IDs, residue_pair_data in get_all_pairs_of_residues_in_motif_MST(motif_MST).items():
        PDBs_that_contain_the_residue_pair = get_PDBs_that_contain_the_residue_pair(pair_of_full_residue_IDs, residue_pair_data, distance_delta_thr, angle_delta_thr, index_folder_path, compression)
        
        if map_of_PDBs_with_all_residue_pairs_initialized_flag == False:
            map_of_PDBs_with_all_residue_pairs = defaultdict(list, PDBs_that_contain_the_residue_pair)
            map_of_PDBs_with_all_residue_pairs_initialized_flag = True
            visited_nodes_map = {
                PDB_ID:set(flatten_iterable(list_of_residue_pairs))
                for PDB_ID, list_of_residue_pairs in map_of_PDBs_with_all_residue_pairs.items()
            }

        else:
            update_map_of_PDBs_with_all_residue_pairs(
                map_of_PDBs_with_all_residue_pairs, 
                PDBs_that_contain_the_residue_pair,
                visited_nodes_map
            )

    # Non-parallel version of filter_out_PDBs_with_unconnected_residue_pairs. Using while loop + popitem()
    # instead of a for loop to limit RAM usage.
    add_resname_as_node_attribute(motif_MST) # Needed for subgraph monomorphism, see run_subgraph_monomorphism().
    filtered_PDBs_with_all_residue_pairs: Dict[str, List[nx.Graph]] =  {}
    while map_of_PDBs_with_all_residue_pairs:
        PDB_ID, pairs_of_residues = map_of_PDBs_with_all_residue_pairs.popitem()
        monomorphism_checked_motifs = run_subgraph_monomorphism(motif_MST, pairs_of_residues)
        
        if not monomorphism_checked_motifs: # These are the false positive PDBs, i.e PDBs that have all the residue pairs but where the monomorphism check doesn't find any similar motif
            continue

        filtered_PDBs_with_all_residue_pairs[PDB_ID] = monomorphism_checked_motifs

    return filtered_PDBs_with_all_residue_pairs

def get_tqdm_progress_bar(total: int, desc: str) -> tqdm:
    return tqdm(total=total, desc=desc, position=0, leave=True, smoothing=0, bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt} | Ellapsed={elapsed}; Remaining={remaining} |')

def get_PDBs_with_similar_motifs(
        reference_motif_MST: nx.Graph, motif_residues_data: Dict[str, Residue], index_folder_path: Path, max_n_mutated_residues: int, 
        residue_type_policy: Union[str, Dict[str,List[str]]], distance_delta_thr: float, angle_delta_thr: float, compression: str, n_cores: int
    ) -> Dict[nx.Graph, Dict[str, List[nx.Graph]]]:
    """
    ...
    """
    motif_MST_filtered_PDBs_map: Dict[nx.Graph, Dict[str, List[nx.Graph]]] = {}
    n_motifs_to_solve = sum(1 for _ in get_all_motif_MSTs_generator(reference_motif_MST, max_n_mutated_residues, residue_type_policy, motif_residues_data))

    with Parallel(n_jobs=n_cores, return_as='generator') as parallel:
        # Residue-pair level parallelisation
        if n_motifs_to_solve <= n_cores:
            for motif_MST in get_all_motif_MSTs_generator(reference_motif_MST, max_n_mutated_residues, residue_type_policy, motif_residues_data):
                PDBs_with_all_residue_pairs = find_PDBs_with_all_residue_pairs(motif_MST, index_folder_path, distance_delta_thr, angle_delta_thr, compression, parallel)
                filtered_PDBs_with_all_residue_pairs = filter_out_PDBs_with_unconnected_residue_pairs(PDBs_with_all_residue_pairs, motif_MST, parallel)

                if filtered_PDBs_with_all_residue_pairs:
                    motif_MST_filtered_PDBs_map[motif_MST] = filtered_PDBs_with_all_residue_pairs
        
        # Motif level parallelisation. When having to check a large number of motifs as a result of mutated residues provided 
        # by the user, motif level parallelisation is ~35% faster than residue-pair level parallelisation.
        else:
            all_motif_MSTs_generator = get_all_motif_MSTs_generator(reference_motif_MST, max_n_mutated_residues, residue_type_policy, motif_residues_data)
            delayed_func: Callable[[nx.Graph, Path, float, float, str], Dict[str, List[nx.Graph]]] = delayed(solve_motif_MST)
            results_generator: Iterator[Dict[str, List[nx.Graph]]] = parallel(
                delayed_func(motif_MST, index_folder_path, distance_delta_thr, angle_delta_thr, compression) 
                for motif_MST in all_motif_MSTs_generator
            )

            all_motif_MSTs_generator = get_all_motif_MSTs_generator(reference_motif_MST, max_n_mutated_residues, residue_type_policy, motif_residues_data)
            tqdm_progress_bar = get_tqdm_progress_bar(total=n_motifs_to_solve, desc='Searched motifs')
            for motif_MST, filtered_PDBs_with_all_residue_pairs in zip(all_motif_MSTs_generator, results_generator, strict=True):
                if filtered_PDBs_with_all_residue_pairs:
                    motif_MST_filtered_PDBs_map[motif_MST] = filtered_PDBs_with_all_residue_pairs
                
                tqdm_progress_bar.update()

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
    
    motif_residues_data = extract_motif_residues_from_PDB_file(PDB_file, motif)
    data_of_all_pairs_of_residues_in_motif = get_data_of_all_pairs_of_residues_in_motif(motif_residues_data)

    # It would be inefficient to search the index for every single pair of residues in the motif in order to find PDBs that have a similar 
    # motif. Instead we can determine the Minimum Spanning Tree (MST) of the motif to limit the search to a subset of pairs that covers all the residues.
    motif_MST = get_minimum_spanning_tree(data_of_all_pairs_of_residues_in_motif)

    compression = detect_the_compression_algorithm_used_in_the_index(index_folder_path)
    
    PDBs_with_similar_motifs = get_PDBs_with_similar_motifs(
        motif_MST, 
        motif_residues_data, 
        index_folder_path,
        max_n_mutated_residues, 
        residue_type_policy,
        distance_delta_thr, 
        angle_delta_thr,
        compression,
        n_cores
    )

    return motif_MST, PDBs_with_similar_motifs
