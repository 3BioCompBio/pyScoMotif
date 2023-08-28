# Todo list

- Could stop the solve_motif function earlier by checking if the current list of PDBs with all the pairs is empty, in which case we can stop computing the remaining pairs.
- The motif search speed could be improved when dealing with many mutated motifs, see motif_search_with_ShareableList.py. Another way to speed it up would be to optimize the subgraph monomorphism code, either by using a faster library implemented in C/C++ or a Cython version of NetworkX. A detailed profiling of the motif search code is needed to see where the bottlenecks are.
- Currently we don't recalculate the sidechain CMR coordinates of mutated residues in the reference motif, although ideally we should do it. We also might want to review how we deal with the special case of Glycine.