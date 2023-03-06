# pyScoMotif: Discovery of similar 3D structural motifs across proteins

## Description

pyScoMotif is a Python package that enables the rapid search of similar 3D structural motifs across large sets of proteins. Concretely, given a reference set of residues in a PDB file, it will find other protein structures with the same set of residues that are in a similar geometrical arragement. 

Typical use cases include the search of catalytic and binding sites.

Reference publication: xx xx

**LICENSE: pyScoMotif is free to use for non-commercial purposes. To use pyScoMotif for commercial purposes, please contact us.**

## Install
To install, simply run:

```
pip install pyscomotif
```
If the install was successful, you should be able to type and run `pyscomotif` in your terminal.

## Tutorial

This tutorial shows how to use pyScoMotif, as well as some more advanced options such as [mutated motifs and position specific exchanges](###searching-for-similar-motifs-with-potential-mutations). A detailed explanation of each parameter available for the different commands can be displayed by typing `pyscomotif <command name> --help`.

### Pre-built indexes

To use pyScoMotif to search for similar motifs, we first need an index of the set of PDB files on which we want to perform the search. Given indexing large sets of PDB files such as the entire Protein Data Bank or AlphaFold2 (AF2) proteomes can take some time, pre-built indexes are available for download for the entire PDB, the AF2 global health proteomes and the AF2 human proteome at http://babylone.ulb.ac.be/pyScoMotif/data/.

### Creating an index

If our pre-built indexes don't cover the PDBs you are interested in, then you will have to build an index yourself (don't panic, it's easy !). For that, we use the `create-index` command, which has 2 mandatory parameters:
- The full path of the directory that contains our PDB files, which is given as the final argument of the command.
- The file extension pattern of the PDB files, which can be specified with the `--pattern` option. 

Additionaly, we can specify the path of the directory that will contain all the index files through the `--index_path` option, and we can also parallelize the indexing through the `--n_cores` option.

Suppose we have downloaded the human proteome dataset from [AlphaFoldDB](https://alphafold.ebi.ac.uk/download), our command would be

```
pyscomotif create-index --pattern=*.pdb.gz --n_cores=6 /home/user/Downloads/UP000005640_9606_HUMAN_v3
```

To update an index with new PDB files, see the `pyscomotif update-index --help` command.

### Searching for similar motifs
Once we have the index of our set of PDB structures, we can perform the similar motif search. 

We will showcase the search of the [serine protease catalytic site](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/173/), which is a 3 residue motif (His-Asp-Ser) that catalyzes the proteolyses of peptide bonds.
For that, we use the `motif-search` command, which has 3 mandatory parameters:
- The full path of the index directory, which by default is created inside our database folder and is named 'pyScoMotif_index'.
- The path of the PDB file that contains our motif of interest.
- The list of residue identifiers of our motif.

Additionally, we can control the distance and angle tolerance values when searching for similar pairs of residues through the `--distance_delta_thr` and `--angle_delta_thr` options, and can also control the maximum RMSD of the hits returned by pyScoMotif through the `--RMSD_threshold` option. Search can also be parallelized through the `--n_cores` option. Finally, we can specify the path of the output csv file that will contain the results through the `--results_output_path` option.

Here we use PDB 1pq5 and residues A56, A99 and A195 as our reference motif, so our search command would be
```
pyscomotif motif-search --results_output_path=/home/user/Downloads/serine_protease_pyScoMotif_result.csv --n_cores=6 /home/user/Downloads/UP000005640_9606_HUMAN_v3/pyScoMotif_index /home/user/Downloads/1pq5.pdb A56 A99 A195
```

This generates the following output table (first 5 results only)

| | **matched_motif** | **similar_motif_found** | **RMSD** | **n_mutations** | **PDB_ID**            | **header_description**                                                           |
|------|-------------------|-------------------------|----------|-----------------|-----------------------|----------------------------------------------------------------------------------|
| 0    | A56H A99D A195S   | A225H A270D A366S       | 0.129    | 0               | AF-Q86T26-F1-model_v3 | alphafold monomer v2.0 prediction for transmembrane protease serine 11b (q86t26) |
| 1    | A56H A99D A195S   | A74H A122D A218S        | 0.134    | 0               | AF-Q6UWY2-F1-model_v3 | alphafold monomer v2.0 prediction for serine protease 57 (q6uwy2)                |
| 2    | A56H A99D A195S   | A70H A112D A205S        | 0.144    | 0               | AF-P49862-F1-model_v3 | alphafold monomer v2.0 prediction for kallikrein-7 (p49862)                      |
| 3    | A56H A99D A195S   | A357H A406D A513S       | 0.144    | 0               | AF-P00750-F1-model_v3 | alphafold monomer v2.0 prediction for tissue-type plasminogen activator (p00750) |
| 4    | A56H A99D A195S   | A70H A117D A202S        | 0.149    | 0               | AF-P08246-F1-model_v3 | alphafold monomer v2.0 prediction for neutrophil elastase (p08246)               |

**matched_motif**: Reference motif residue IDs that were matched. 
**similar_motif_found**: Residue IDs of the similar motif that was found. 
**RMSD**: Root Mean Square Deviation between the reference and similar motif found.
**n_mutations**: Number of mutated residues relative to the original reference motif.
**PDB_ID**: PDB ID that contains the similar motif that was found.    
**header_description**: Text description of the PDB file that contains the similar motif found.

### Searching for similar motifs with potential mutations

The serine protease catalytic site example is simple in the sense that we only want to find PDB structures with motifs that exactly match the residue types of the reference motif (i.e Histidine, Aspartate and Serine). But in some cases the constraints may be more relaxed for certain residues, in which case one needs to also search for mutated versions of the motif. pyScoMotif's `motif-search` command comes with this capability built-in.

In this section we showcase the search of the [alcohol dehydrogenase](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/256/) catalytic site, which can tolerate different mutations at different positions.

There are two options that control the search for mutated versions of a reference motif:
- The `--residue_type_policy` option, for which there are 4 possibilities:
    - "strict" (default): no mutations allowed, only search for motifs that have <ins>exactly</ins> the same residue types as the reference motif.
    - "relaxed": residues can mutate according to their residue type group, which are: non-polar (GAVLI), polar (STPNQ), sulfur (MC), positives (KRH), negatives (DE), aromatic (FYW).
    - "fully_relaxed": residues can be mutated to any of the other 19 possible residues.
    - A custom position specific exchange in JSON format. This allows complete control over the possible mutations of each residue. The format takes residue IDs as keys and a string of possible mutations as values (e.g: '{"A1":"KR", "A2":"YW", "A5":"D"}'. Note the use of single and double quotes).

- The `--max_n_mutated_residues` option, which controls the maximum number of combinations of mutations that should be allowed when generating mutated motifs. This option can be very useful if you want to search for similar motifs carrying multiple mutations. It is set to 1 by default.

Here we use PDB 1hso and residues A49, A60, A102, A175, A183, A233 and A235 as our reference motif, and take advantage of the fully_relaxed option to search for all the possible mutated versions carrying 1 mutation, so our search command would be:

```
pyscomotif motif-search --residue_type_policy=fully_relaxed --results_output_path=/home/user/Downloads/alcohol_dehydrogenase_pyScoMotif_result.csv --n_cores=6 /home/user/Downloads/UP000005640_9606_HUMAN_v3/pyScoMotif_index /home/user/Downloads/1hso.pdb A49 A60 A102 A175 A183 A233 A235
```