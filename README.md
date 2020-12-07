# structured-attention-graphs


## Abstract
Attention maps are popular tools of explaining the decisions of convolutional networks for image classification. Typically for each image of interest, a single attention map is produced, which  assigns  weights  to  pixels  based  on  their  importance to  the  classification.  We  argue  that  a  single  attention  map provides an incomplete understanding since there are often many other maps that explain a classification equally well. In this paper, we introduce *structured attention graphs (SAGs)*, which compactly represent sets of attention maps for an image  by  capturing  how  different  combinations  of  image  regions impact the confidence of a classifier. We propose an approach to compute SAGs and a visualization for SAGs so that deeper insight can be gained into a classifier’s decisions.
[arxiv link][]

## Demo

<img src="demo_images/peacock_original.png" width="50" height="50">  |   <img src="demo_images/peacock_gcam.png" width="50" height="50"> | <img src="demo_images/peacock_igos.png" width="50" height="50"> | <img src="demo_images/peacock_dnf.png" width="50" height="50">

<img src="demo_images/peacock_sag.png" width="50" height="50">


### To run:
`python main_generate_sag.py`

### Directory description:
- *combinatorial_search.py* -- functions to perform combinatorial search on a perturbation mask
- *diverse_subset_selection.py* -- functions to obtain a diverse subset from a set of candidate masks
- *get_perturbation_mask.py* -- functions to obtain a perturbation mask
- *patch_deletion_tree.py* -- functions to build a tree by deleting one patch at a time
- *utils.py* -- helper functions
- *main_generate_sag.py* -- main file to generate SAG
- *Images* -- folder consisting of input images for which we build SAGs
- *Results* -- all results are populated here
- *GroundTruth1000.txt* -- IMAGENET groundtruth labels
- *requirements.txt* -- lists all libraries required to run the code
