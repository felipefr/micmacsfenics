#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 17:36:14 2025

@author: frocha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 23:30:58 2024

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2025, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or <felipe.f.rocha@gmail.com>
"""

"""
DESCRIPTION: 
    Computational homogenisation for nonlinear materials (infinitesimal strain).
    It computes both homogenised stress and tangent tensors.
    
CONVENTIONS:
    Infinitesimal (Linear) strain, but nonlinear constitutive
    Mandel notation throughout
    Green-Elastic material (strain energy density potential)
    Split between affine (given) and non-affine (fluctuations) parts of displacement
    
KNOWN LIMITATIONS:
    Linear boundary (zero boundary fluctuations) implemented
     
"""

import sys
import numpy as np
from timeit import default_timer as timer
from functools import partial 

import basix
import numpy as np
from dolfinx import fem, io
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
import fetricksx as ft
import fetricksx.mechanics.conversions3d as conv3d
import fetricksx.mechanics.conversions as conv2d  

class MicroModelFiniteStrain:

    def __init__(self, mesh, psi_mu, bnd_flags, solver_param=None):

        self.mesh = mesh
        self.solver_param = solver_param
        self.tdim = self.mesh.topology.dim
        self.tensor_encoding = "unsym"
        
        self.conv = {2: conv2d, 3: conv3d}[self.tdim]
        if(self.tensor_encoding == "unsym"):
            self.nstrain = self.tdim*self.tdim
            self.grad = self.conv.grad_unsym
            
        elif(self.tensor_encoding == "mandel"):
            self.nstrain = int(self.tdim*(self.tdim + 1)/2)
            self.grad = self.conv.symgrad_mandel
            
        self.psi_mu = psi_mu
        self.Uh = fem.functionspace(self.mesh, ("CG", self.solver_param['poly_order'], (self.tdim,)))
        uD = np.zeros(self.tdim, dtype=ScalarType)
        
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
        self.Gmacro = fem.Constant(self.mesh, np.zeros(self.nstrain, dtype=ScalarType))  # just placeholder
        self.Gmacro_kl = fem.Constant(self.mesh, np.zeros(self.nstrain, dtype=ScalarType))  # just placeholder
        
        self.duh = ufl.TrialFunction(self.Uh)            # Incremental displacement
        self.vh  = ufl.TestFunction(self.Uh)             # Test function
        self.uh  = fem.Function(self.Uh)                 # Displacement from previous iteration
        self.ukl = fem.Function(self.Uh)
        
    def set_microproblem(self):
        
        dy, Gmacro = self.dy, self.Gmacro
        uh, vh, duh = self.uh, self.vh, self.duh
        
        Fmu = self.conv.Id_unsym_df + Gmacro + self.grad(uh)
        Fmu_var = ufl.variable(Fmu)
        Fmu_ = self.conv.unsym2tensor(Fmu_var)
        
        self.PKmu, self.Amu = ft.get_stress_tang_from_psi(self.psi_mu, Fmu_, Fmu_var) 
        
        self.Jac = ufl.inner(ufl.dot( self.Amu, self.grad(duh)), self.grad(vh))*dy
        
        # generic residual for either fluctuation and canonical problem
        self.flag_linres = fem.Constant(self.mesh, 0.0)
        self.flag_nonlinres = fem.Constant(self.mesh, 1.0)
        r = self.flag_nonlinres*self.PKmu + self.flag_linres*ufl.dot( self.Amu, self.Gmacro_kl)
        
        self.Res = ufl.inner(r , self.grad(vh))*dy
                
        self.microproblem = ft.CustomNonlinearProblem(self.Res, uh, self.bcD, self.Jac)
        self.microsolver = ft.CustomNonlinearSolver(self.microproblem)
    
    def get_stress_tangent(self):
        return self.homogenise_stress(), ft.sym_flatten_3x3_np(self.homogenise_tangent())
    
    def get_stress_tangent_solve(self, e):
        self.solve_microproblem(e)
        return np.concatenate((self.homogenise_stress(), 
                               ft.sym_flatten_3x3_np(self.homogenise_tangent())))
    
    def get_stress(self, e = None, t=None):
        if(e is not None):
            self.solve_microproblem(e)
        if(t is not None):
            t = self.homogenise_tangent()
        
        return self.homogenise_stress()

    def homogenise_tangent(self):
        self.flag_nonlinres.value = 0.0
        self.flag_linres.value = 1.0
        tangenthom = np.zeros((self.nstrain,self.nstrain))
        stress_kl = ufl.dot(self.Amu, self.Gmacro_kl +  self.grad(self.microsolver.tangent_problem.u))
        
        for i in range(self.nstrain):
            self.Gmacro_kl.value[i] = 1.0
            self.microsolver.tangent_problem.assemble_rhs()
            self.microsolver.tangent_problem.assemble_lhs() # comment for fastness, but loses precision
            self.microsolver.tangent_problem.solve_system()
            tangenthom[i,:] = self.inv_vol*ft.integral(stress_kl, self.dy, self.mesh, ((self.nstrain,)))
            self.Gmacro_kl.value[i] = 0.0
        
        return tangenthom
    
    def homogenise_stress(self):
        return self.inv_vol*ft.integral(self.PKmu, self.dy, self.mesh, (self.nstrain,))
        
    def solve_microproblem(self, e):
        self.flag_nonlinres.value = 1.0
        self.flag_linres.value = 0.0
        # self.restart_initial_guess()
        self.Gmacro.value[:] = e[:]
        self.microsolver.solve()
        

        
