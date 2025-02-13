# micmacsfenics
FEniCSx-based implementation for multi-scale problems (FE2) and computational homogenisation. 

Currently, micmacsfenics is going through a complete factorization from the original library in legacy FEniCS. Please check the micmacsfenics_legacy branch for working version of the codes. The main changes will be:
- Unique way of doing things. Maybe too general, but unique.
- Use fetricks (https://github.com/felipefr/fetricks) as much as possible for general things, to keep micmacsfenics as compact as possible.
- Inelasticty using Mfront or dolfinx-external-operator (https://github.com/a-latyshev/)
- Implementation of the generalised minimally contraint bc (https://doi.org/10.1016/j.ijsolstr.2023.112494)
- Be the most flexible as possible concerning physics. 
- Implementation of non-standard types of boundary conditions: periodic, minimally constraint in FEniCSx (in legacy FEniCS this was already done).

## Installation
- conda create -n fenicsx-env -c conda-forge fenics-dolfinx mpich pyvista scipy
- conda activate fenicsx-env
- pip install gmsh meshio pygmsh
- conda install spyder-kernels (only if spyder is used as IDE)

Currently, micmacsfenicsx run with the versions: fenics-dolfinx 0.9.0, gmsh 4.13.1, meshio 5.3.5, pygmsh 7.1.17

## Citing 
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
