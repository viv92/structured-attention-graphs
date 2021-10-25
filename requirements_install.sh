#!/bin/bash
conda create -n structured_attention_graphs python=3
conda activate structured_attention_graphs
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c anaconda scipy
conda install -c conda-forge matplotlib
conda install scikit-image
conda install -c conda-forge pygraphviz
pip install opencv-python
