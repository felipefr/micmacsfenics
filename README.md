# micmacsfenics
FEniCSx-based implementation for multi-scale problems (FE2) and computational homogenisation using FEniCS 0.10. Currently, micmacsfenicsx is going through a complete factorization from the original library in legacy FEniCS and codes Fenicsx 0.9 (not fully functional). Please check the fenics_legacy branch for working version of the codes. The main changes will be:
- Unique way of doing things. Maybe too general, but unique.
- Rely in in a local subset of fetricks (https://github.com/felipefr/fetricks) to keep it as self-contained as possible.
- Rely on https://bleyerj.github.io/dolfinx_materials/ for inelastic problems.
- Implementation of the generalised minimally contraint bc (https://doi.org/10.1016/j.ijsolstr.2023.112494) using the Real Space, absent in FenicsX, but available in https://github.com/scientificcomputing/scifem.
- Implementation of non-standard types of boundary conditions: periodic, minimally constraint in FEniCSx using dolfin-mpc (http://jsdokken.com/dolfinx_mpc/README.html) (in legacy FEniCS this was already done).
- Be the most flexible as possible concerning physics. 

## Installation
- conda create -n fenicsx-env -c conda-forge fenics-dolfinx mpich pyvista scipy
- conda activate fenicsx-env
- pip install gmsh meshio pygmsh
- conda install spyder-kernels (only if spyder is used as IDE)

Currently, micmacsfenicsx run with the versions: fenics-dolfinx 0.10.0, gmsh, meshio, pygmsh (versions 4.13.1, 5.3.5, 7.1.17, but upgrating them)

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
