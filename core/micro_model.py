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


solver_parameters = {"nonlinear_solver": "newton",
                     "newton_solver": {"maximum_iterations": 20,
                                       "report": False,
                                       "error_on_nonconvergence": True}}

class MicroModel:

    def __init__(self, mesh, psi_mu, bnd_flags):

        self.mesh = mesh
        self.tdim = self.mesh.topology.dim
        self.nmandel = int(self.tdim*(self.tdim + 1)/2)        
        
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
        
        self.microproblem = ufl.NonlinearVariationalProblem(self.Res, uh, self.bcD, self.Jac)
        self.microsolver = ufl.NonlinearVariationalSolver(self.microproblem)
        self.microsolver.parameters.update(solver_parameters)
    
    
    def setCanonicalproblem(self):
        dy, vh = self.dy, self.vh
                 
        # negative because 
        self.RHS_can = -ufl.inner(ufl.dot( self.Cmu, self.Eps_kl), ft.symgrad_mandel(vh))*dy 
        
        self.Acan = ufl.PETScMatrix()
        self.bcan = ufl.PETScVector()
        self.solver_can = ufl.PETScLUSolver()
        
    
    def __homogeniseTangent(self):
        
        # print("index" , epsMacro)
    
        dy, vol, Eps_kl, Cmu, ukl = self.dy, self.vol, self.Eps_kl, self.Cmu, self.ukl
        
        ufl.assemble(self.Jac, tensor = self.Acan)
        self.bcD.apply(self.Acan)
                        
        self.tangenthom.fill(0.0)
        
        unit_vec = np.zeros(self.nmandel)
        
        for i in range(self.nmandel):
            
            unit_vec[i] = 1.0
            Eps_kl.assign(fem.Constant(self.mesh, unit_vec))
            
            ufl.assemble(self.RHS_can, tensor = self.bcan)
            self.bcD.apply(self.bcan)
    
            self.solver_can.solve(self.Acan, ukl.vector(), self.bcan)
        
            self.tangenthom[i,:] += ft.Integral(ufl.dot(Cmu, Eps_kl +  ft.symgrad_mandel(ukl)) , dy, ((self.nmandel,)))/vol
            
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
        
