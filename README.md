# micmacsFenics
Implementations using Fenics and Multiphenics to solve Computational Homogenisation problems. 

This encompasses the implementation of non-standard types of boundary conditions, indeed general constraints. I also made use of the Multiphenics, that allows a simple of implementation for problems of multiples variables. In this case, we have the main variable and auxiliary Lagrange multipliers variables defined in some parts of the domain that encodes the variational equation that enforces the constraint weakly. 

In this branch we aim to provide a general setting to include nonlinear analysisincluding inelasticity, but so far solving the problems in hyperlastic (but small strain scenario)
 
If this library has been useful for you, please the article in which this library is related with:

@article{rocha2021deepbnd,
  title={Deepbnd: a machine learning approach to enhance multiscale solid mechanics},
  author={Rocha, Felipe and Deparis, Simone and Antolin, Pablo and Buffa, Annalisa},
  journal={arXiv preprint arXiv:2110.11141},
  year={2021}
}

[![DOI](https://zenodo.org/badge/341954015.svg)](https://zenodo.org/badge/latestdoi/341954015)
