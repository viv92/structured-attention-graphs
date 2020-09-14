# structured-attention-graphs
Source code to generate SAGs [paper link]

### To run:
python main_generate_sag.py

### Directory structure:
combinatorial_search.py -- functions to perform combinatorial search on a perturbation mask
diverse_subset_selection.py -- functions to obtain a diverse subset from a set of candidate masks
get_perturbation_mask.py -- functions to obtain a perturbation mask
patch_deletion_tree.py -- functions to build a tree by deleting one patch at a time
utils.py -- helper functions
main_generate_sag.py -- main file to generate SAG
Images -- folder consisting of input images for which we build SAGs
Results -- all results are populated here
GroundTruth1000.txt -- IMAGENET groundtruth labels
requirements.txt -- lists all libraries required to run the code 
