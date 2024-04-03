# micmacsFenics
Implementations using Fenics and Multiphenics to solve Computational Homogenisation problems. 

This encompasses the implementation of non-standard types of boundary conditions, indeed general constraints. I also made use of the Multiphenics, that allows a simple of implementation for problems of multiples variables. In this case, we have the main variable and auxiliary Lagrange multipliers variables defined in some parts of the domain that encodes the variational equation that enforces the constraint weakly. 

In this branch we aim to provide a general setting to include nonlinear analysisincluding inelasticity, but so far solving the problems in hyperlastic (but small strain scenario)
 
# Instalation
If you want to keep track of files for uninstall reasons, do:
python setup.py install --record files.txt
xargs rm -rf < files.txt

If this library has been useful for you, please the article in which this library is related with:

@article{Rocha2023,
title = {DeepBND: A machine learning approach to enhance multiscale solid mechanics},
journal = {Journal of Computational Physics},
pages = {111996},
year = {2023},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2023.111996},
url = {https://www.sciencedirect.com/science/article/pii/S0021999123000918},
author = {Felipe Rocha and Simone Deparis and Pablo Antolin and Annalisa Buffa}
}

[![DOI](https://zenodo.org/badge/341954015.svg)](https://zenodo.org/badge/latestdoi/341954015)
