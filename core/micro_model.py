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
        
        self.getStress = self.__computeStress
        self.getTangent = self.__computeTangent

        self.declareAuxVariables()
        self.setMicroproblem()
        self.setCanonicalproblem()   
    
    def restart_initial_guess(self):
        self.uh.vector().set_local(np.zeros(self.Uh.dim()))
        
    def getStressTangent(self, e):
        # return np.concatenate((self.getStress(e), symflatten(self.getTangent(e))))
        return self.getStress(e), ft.sym_flatten_3x3_np(self.getTangent(e)) # already in the mandel format

    def getStressTangent_force(self, e): # force to recompute
        self.setUpdateFlag(False)
        return self.getStressTangent(e)
    
        
    def declareAuxVariables(self):
    
        self.dy = ufl.Measure('dx', self.mesh)
        # self.vol = ufl.assemble(ufl.Constant(1.0)*self.dy)
        self.vol = 1.0 # todo commet above
        self.y = ufl.SpatialCoordinate(self.mesh)
        self.Eps_array = np.array([0., 0., 0.], dtype = ScalarType)
        self.Eps_kl_array = np.array([0., 0., 0.], dtype = ScalarType)
        self.Eps = fem.Constant(self.mesh, self.Eps_array)  # just placeholder
        self.Eps_kl = fem.Constant(self.mesh, self.Eps_kl_array)  # just placeholder
        
        self.duh = ufl.TrialFunction(self.Uh)            # Incremental displacement
        self.vh  = ufl.TestFunction(self.Uh)             # Test function
        self.uh  = fem.Function(self.Uh)                 # Displacement from previous iteration
        self.ukl = fem.Function(self.Uh)
        
        self.stresshom = np.zeros(self.nmandel)
        self.tangenthom = np.zeros((self.nmandel,self.nmandel))
        
    def setMicroproblem(self):
        
        dy, Eps = self.dy, self.Eps
        uh, vh, duh = self.uh, self.vh, self.duh
        
        eps = Eps  + ft.symgrad_mandel(uh)            # Deformation gradient
        eps_var = ufl.variable(eps)
        
        self.sigmu, self.Cmu = ft.get_stress_tang_from_psi(self.psi_mu, eps_var, eps_var) 
        
        self.Res = ufl.inner(self.sigmu , ft.symgrad_mandel(vh))*dy 
        self.Jac = ufl.inner(ufl.dot( self.Cmu, ft.symgrad_mandel(duh)), ft.symgrad_mandel(vh))*dy
        
    
        self.microproblem = fem.petsc.NonlinearProblem(self.Res, uh, self.bcD, self.Jac)
        self.microsolver = dolfinx.nls.petsc.NewtonSolver(self.mesh.comm, self.microproblem)
        self.microsolver.atol = self.solver_param['atol']
        self.microsolver.rtol = self.solver_param['rtol']    
    
    def setCanonicalproblem(self):
        dy, vh = self.dy, self.vh
        self.RHS_can = -ufl.inner(ufl.dot( self.Cmu, self.Eps_kl), ft.symgrad_mandel(vh))*dy 
        self.solver_can = ft.CustomLinearSolver(self.Jac, self.RHS_can, self.ukl, self.bcD)
        
    def __homogeniseTangent(self):
        self.solver_can.assembly_lhs()
        self.tangenthom.fill(0.0)
        
        unit_vec = np.zeros(self.nmandel)
        
        for i in range(self.nmandel):
            
            unit_vec[i] = 1.0
            self.Eps_kl.assign(fem.Constant(self.mesh, unit_vec))
            
            self.solver_can.assembly_rhs()
            self.solver_can.solve()
        
            self.tangenthom[i,:] += (1.0/self.vol)*ft.Integral(
                ufl.dot(self.Cmu, self.Eps_kl +  ft.symgrad_mandel(self.ukl)) , 
                self.dy, ((self.nmandel,)))
            
            unit_vec[i] = 0.0
            
    def __homogeniseStress(self):
        self.stresshom = ft.Integral(self.sigmu, self.dy, (self.nmandel,))/self.vol
        
    def __computeFluctuations(self, e):
        self.restart_initial_guess()
        self.Eps.assign(fem.Constant(self.mesh, e))
        self.microsolver.solve()
        
    def __computeStress(self, e):
        if(not self.isFluctuationUpdated):
            self.__computeFluctuations(e)
    
        self.__homogeniseStress()
    
        return self.__returnStress(e)
        
    
    def __computeTangent(self, e):
        
        if(not self.isFluctuationUpdated):
            self.__computeFluctuations(e)
            
        self.__homogeniseTangent()
        
        return self.__returnTangent(e)
        
