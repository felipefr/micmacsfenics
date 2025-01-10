#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 23:59:30 2022

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
import multiphenics as mp
import dolfin as df
from timeit import default_timer as timer
from ufl import nabla_div, indices
from functools import partial 

import fetricks as ft

# wrt respect to the nonlinear version, the optmised adds a global flag instead of local strains

solver_parameters = {"nonlinear_solver": "newton",
                     "newton_solver": {"maximum_iterations": 20,
                                       "report": False,
                                       "error_on_nonconvergence": True}}

ind_sym_tensor = np.array([0, 1, 2, 4, 5, 8])

def getPsi(e, param):
    tr_e = ft.tr_mandel(e)
    e2 = df.inner(e, e)

    lamb, mu, alpha = param
    
    return (0.5*lamb*(1.0 + 0.5*alpha*(tr_e**2))*(tr_e**2) + mu*(1 + 0.5*alpha*e2)*(e2))

class MicroConstitutiveModelNonlinearOpt: # TODO derive it again from a base class

    # Counter of calls 
    countComputeFluctuations = 0
    countComputeCanonicalProblem = 0
   
    def __init__(self, mesh, param, model = None):

        self.mesh = mesh
        self.ndim = 2
        self.nvoigt = int(self.ndim*(self.ndim + 1)/2)
        self.tensor_encoding = "mandel"        
        
        self.param = param
        
        self.onBoundary = df.CompiledSubDomain('on_boundary')
        self.Uh = df.VectorFunctionSpace(self.mesh, "CG", 1)     
        self.bcD = df.DirichletBC(self.Uh, df.Constant((0, 0)), self.onBoundary)   
        
        self.isStressUpdated = False
        self.isTangentUpdated = False
        self.isFluctuationUpdated = False
        
        self.getStress = self.__computeStress
        self.getTangent = self.__computeTangent

        self.declareAuxVariables()
        self.setMicroproblem()
        self.setCanonicalproblem()
    
    
    def setGlobalVectors(self , eps_global, sig_global, tan_global, id_global, id_global_tan):
        self.eps_global = eps_global
        self.sig_global = sig_global
        self.tan_global = tan_global
        self.id_global = id_global
        self.id_global_tan = id_global_tan
        

    def setUpdateFlag(self, flag):
        self.setStressUpdateFlag(flag)
        self.setTangentUpdateFlag(flag)
        self.setFluctuationUpdateFlag(flag)
        
    def setStressUpdateFlag(self, flag):
        if(flag):
            self.getStress = lambda e: self.stresshom
        else:
            self.getStress = self.__computeStress

    def setTangentUpdateFlag(self, flag):
        if(flag):
            self.getTangent = lambda e: self.tangenthom
        else:
            self.getTangent = self.__computeTangent
    
    def setFluctuationUpdateFlag(self, flag):
        self.isFluctuationUpdated = flag
        
    def declareAuxVariables(self):
    
        self.dy = df.Measure('dx', self.mesh)
        self.vol = df.assemble(df.Constant(1.0)*self.dy)
        self.y = df.SpatialCoordinate(self.mesh)
        self.Eps = df.Constant((0., 0., 0.))  # just placeholder
        self.Eps_kl = df.Constant((0., 0., 0.))  # just placeholder
        
        self.duh = df.TrialFunction(self.Uh)            # Incremental displacement
        self.vh  = df.TestFunction(self.Uh)             # Test function
        self.uh  = df.Function(self.Uh)                 # Displacement from previous iteration
        self.ukl = df.Function(self.Uh)
        
        self.stresshom = np.zeros(self.nvoigt)
        self.tangenthom = np.zeros((self.nvoigt,self.nvoigt))
        
    def setMicroproblem(self):
        
        dy, Eps = self.dy, self.Eps
        uh, vh, duh = self.uh, self.vh, self.duh
        
        eps = Eps  + ft.symgrad_mandel(uh)            # Deformation gradient
        eps_var = df.variable(eps)
        
        psi_mu = getPsi(eps_var, self.param)
        
        self.sigmu = df.diff(psi_mu , eps_var)
        self.Cmu = df.diff(self.sigmu, eps_var)

        self.Res = df.inner(self.sigmu , ft.symgrad_mandel(vh))*dy 
        self.Jac = df.inner(df.dot( self.Cmu, ft.symgrad_mandel(duh)), ft.symgrad_mandel(vh))*dy
        
        self.microproblem = df.NonlinearVariationalProblem(self.Res, uh, self.bcD, self.Jac)
        self.microsolver = df.NonlinearVariationalSolver(self.microproblem)
        self.microsolver.parameters.update(solver_parameters)
    
    
    def setCanonicalproblem(self):
        dy, vh = self.dy, self.vh
                 
        self.RHS_can = -df.inner(df.dot( self.Cmu, self.Eps_kl), ft.symgrad_mandel(vh))*dy 
        
        self.Acan = df.PETScMatrix()
        self.bcan = df.PETScVector()
        self.solver_can = df.PETScLUSolver()
        
    
    def __homogeniseTangent(self):
        
        # print("index" , epsMacro)
    
        dy, vol, Eps_kl, Cmu, ukl = self.dy, self.vol, self.Eps_kl, self.Cmu, self.ukl
        
        df.assemble(self.Jac, tensor = self.Acan)
        self.bcD.apply(self.Acan)
                        
        self.tangenthom.fill(0.0)
        
        unit_vec = np.zeros(self.nvoigt)
        
        for i in range(self.nvoigt):
            
            unit_vec[i] = 1.0
            Eps_kl.assign(df.Constant(unit_vec))
            
            df.assemble(self.RHS_can, tensor = self.bcan)
            self.bcD.apply(self.bcan)
    
            self.solver_can.solve(self.Acan, ukl.vector(), self.bcan)
        
            self.tangenthom[i,:] += ft.Integral(df.dot(Cmu, Eps_kl +  ft.symgrad_mandel(ukl)) , dy, ((self.nvoigt,)))/vol
            
            unit_vec[i] = 0.0
        
        self.tan_global[self.id_global_tan:self.id_global_tan + 6] = self.tangenthom.flatten()[ind_sym_tensor]
        type(self).countComputeCanonicalProblem = type(self).countComputeCanonicalProblem + 1
        
        return self.tangenthom[i,:]
        
    def __homogeniseStress(self):
        self.sig_global[self.id_global:self.id_global+3] = ft.Integral(self.sigmu, self.dy, (self.nvoigt,))/self.vol
        
        return self.stresshom[:]
    
    def __computeFluctuations(self, e):
        self.Eps.assign(df.Constant(self.eps_global[self.id_global:self.id_global+3]))
        self.microsolver.solve()
        self.setFluctuationUpdateFlag(True)
        
        type(self).countComputeFluctuations = type(self).countComputeFluctuations + 1

        
    def __computeStress(self, e):
        if(not self.isFluctuationUpdated):
            self.__computeFluctuations(e)
    
        self.__homogeniseStress()
        self.setStressUpdateFlag(True)
    
        return self.stresshom
        
    
    def __computeTangent(self, e):
        
        if(not self.isFluctuationUpdated):
            self.__computeFluctuations(e)
            
        self.__homogeniseTangent()
        self.setTangentUpdateFlag(True)            
        
        return self.tangenthom
        
