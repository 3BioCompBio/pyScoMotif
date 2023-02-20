# Todo list

- Could stop the solve_motif function earlier by checking if the current list of PDBs with all the pairs is empty, in which case we can stop computing the remaining pairs.
- Currently we don't recalculate the sidechain CMR coordinates of mutated residues in the reference motif, although ideally we should do it.
- The motif search speed could be improved when dealing with many mutated motifs, see motif_search_with_SareableList.py. Another way to speed it up would be to optimize the subgraph monomorphism code, either by using a faster library implemented in C/C++ or a Cython version of Networkx.