#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 23:30:58 2024

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 20:23:09 2023

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

import sys
import numpy as np
from timeit import default_timer as timer
from functools import partial 

sys.path.append("/home/felipe/sources/fetricksx")

import basix
import numpy as np
from dolfinx import fem, io
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore


import fetricksx as ft


# solver_parameters = {"nonlinear_solver": "newton",
#                      "newton_solver": {"maximum_iterations": 20,
#                                        "report": False,
#                                        "error_on_nonconvergence": True}}

class MicroModel:

    def __init__(self, mesh, psi_mu, bnd_flags, solver_param=None):

        self.mesh = mesh
        self.tdim = self.mesh.topology.dim
        self.nmandel = int(self.tdim*(self.tdim + 1)/2)   
        self.solver_param = solver_param
        
        self.psi_mu = psi_mu
        self.Uh = fem.functionspace(self.mesh, ("CG", 1, (self.tdim,)))
        uD = np.array([0.0,0.0],dtype=ScalarType)
        
        self.bcD = [ ft.dirichletbc(uD, flag, self.Uh) for flag in bnd_flags]
    
        self.set_aux_variables()
        self.set_microproblem()
    
    def restart_initial_guess(self):
        self.uh.x.array[:] = 0.0
        
    def set_aux_variables(self):    
        self.dy = self.mesh.dx
        self.vol = ft.integral(fem.Constant(self.mesh,1.0), self.dy, self.mesh, ())
        self.inv_vol = self.vol**-1
        
        self.y = ufl.SpatialCoordinate(self.mesh)
        self.Eps = fem.Constant(self.mesh, np.array([0., 0., 0.], dtype = ScalarType))  # just placeholder
        self.Eps_kl = fem.Constant(self.mesh, np.array([0., 0., 0.], dtype = ScalarType))  # just placeholder
        
        self.duh = ufl.TrialFunction(self.Uh)            # Incremental displacement
        self.vh  = ufl.TestFunction(self.Uh)             # Test function
        self.uh  = fem.Function(self.Uh)                 # Displacement from previous iteration
        self.ukl = fem.Function(self.Uh)
        
    def set_microproblem(self):
        
        dy, Eps = self.dy, self.Eps
        uh, vh, duh = self.uh, self.vh, self.duh
        
        eps = Eps  + ft.symgrad_mandel(uh)            # Deformation gradient
        eps_var = ufl.variable(eps)
        
        self.sigmu, self.Cmu = ft.get_stress_tang_from_psi(self.psi_mu, eps_var, eps_var) 
        
        self.Jac = ufl.inner(ufl.dot( self.Cmu, ft.symgrad_mandel(duh)), ft.symgrad_mandel(vh))*dy
        
        # generic residual for either fluctuation and canonical problem
        self.flag_linres = fem.Constant(self.mesh, 0.0)
        self.flag_nonlinres = fem.Constant(self.mesh, 1.0)
        r = self.flag_nonlinres*self.sigmu + self.flag_linres*ufl.dot( self.Cmu, self.Eps_kl)
        
        self.Res = ufl.inner(r , ft.symgrad_mandel(vh))*dy
                
        self.microproblem = ft.CustomNonlinearProblem(self.Res, uh, self.bcD, self.Jac)
        self.microsolver = ft.CustomNonlinearSolver(self.microproblem)
    
    def get_stress_tangent(self):
        return self.homogenise_stress(), self.homogenise_tangent()
    
    def get_stress_tangent_solve(self, e):
        self.solve_microproblem(e)
        return np.concatenate((self.homogenise_stress(), self.homogenise_tangent()))
    
    def get_stress(self, e = None, t=None):
        if(e is not None):
            self.solve_microproblem(e)
        if(t is not None):
            t = self.homogenise_tangent()
        
        return self.homogenise_stress()

    def homogenise_tangent(self):
        self.flag_nonlinres.value = 0.0
        self.flag_linres.value = 1.0
        tangenthom = np.zeros((self.nmandel,self.nmandel))
        stress_kl = ufl.dot(self.Cmu, self.Eps_kl +  ft.symgrad_mandel(self.microsolver.tangent_problem.u))
        
        for i in range(self.nmandel):
            self.Eps_kl.value[i] = 1.0
            self.microsolver.tangent_problem.assemble_rhs()
            self.microsolver.tangent_problem.solve_system()
            tangenthom[i,:] = self.inv_vol*ft.integral(stress_kl, self.dy, self.mesh, ((self.nmandel,)))
            self.Eps_kl.value[i] = 0.0
        
        return ft.sym_flatten_3x3_np(tangenthom)
    
    def homogenise_stress(self):
        return self.inv_vol*ft.integral(self.sigmu, self.dy, self.mesh, (self.nmandel,))
        
    def solve_microproblem(self, e):
        self.flag_nonlinres.value = 1.0
        self.flag_linres.value = 0.0
        self.restart_initial_guess()
        self.Eps.value[:] = e[:]
        self.microsolver.solve()
        

        
