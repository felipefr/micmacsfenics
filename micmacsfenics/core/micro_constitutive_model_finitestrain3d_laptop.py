#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:35:31 2025

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
import dolfin as df
from timeit import default_timer as timer
from functools import partial 
import fetricks as ft
import fetricks.mechanics.conversions3d as conv 
import warnings

solver_parameters = {"nonlinear_solver": "newton",
                     "newton_solver": {"maximum_iterations": 20,
                                       "report": False,
                                       "error_on_nonconvergence": True}}

class MicroConstitutiveModelFiniteStrain3d: # TODO derive it again from a base class

    # Counter of calls 
    countComputeFluctuations = 0
    countComputeCanonicalProblem = 0
    countTangentCalls = 0
    countStressCalls = 0
   
    def __init__(self, mesh, psi_mu, bnd_model = None):
        pass
        self.mesh = mesh
        self.ndim = 3 # 3d
        self.tensor_encoding = "unsym"        
        self.nstrain = self.ndim*self.ndim
        
        self.psi_mu = psi_mu
        
        self.Uh = df.VectorFunctionSpace(self.mesh, "CG", 1) 
        
        onBoundary = df.CompiledSubDomain('on_boundary')
        self.bcD = [df.DirichletBC(self.Uh, df.Constant((0,0,0)), onBoundary)]   # 3d
        # self.bcD = [df.DirichletBC(self.Uh, df.Constant((0,0,0)), self.mesh.boundaries, 1)]   # 3d
        
        self.isStressUpdated = False
        self.isTangentUpdated = False
        self.isFluctuationUpdated = False
        
        self.getStress = self.__computeStress
        self.getTangent = self.__computeTangent

        self.declareAuxVariables()
        self.setMicroproblem()
        self.setCanonicalproblem()
    
    
    # seems it is not working
    def restart_counters(self):
        self.countComputeFluctuations = 0
        self.countComputeCanonicalProblem = 0
        self.countTangentCalls = 0
        self.countStressCalls = 0        
    
    def restart_initial_guess(self):
        self.uh.vector().set_local(np.zeros(self.Uh.dim()))
        
    
    def setUpdateFlag(self, flag):
        self.setStressUpdateFlag(flag)
        self.setTangentUpdateFlag(flag)
        self.setFluctuationUpdateFlag(flag)
        
    def setStressUpdateFlag(self, flag):
        if(flag):
            self.getStress = self.__returnStress
        else:
            self.getStress = self.__computeStress

    
    def getStressTangent(self, e):
        return self.getStress(e), ft.sym_flatten_9x9_np(self.getTangent(e)) #3d

    def getStressTangent_force(self, e): # force to recompute
        self.setUpdateFlag(False)
        return self.getStressTangent(e)
    
    def __returnTangent(self, e):
        type(self).countTangentCalls = type(self).countTangentCalls + 1     
        return self.tangenthom
    
    def __returnStress(self, e):
        type(self).countStressCalls = type(self).countStressCalls + 1     
        return self.stresshom
    
    def setTangentUpdateFlag(self, flag):
        if(flag):
            self.getTangent = self.__returnTangent 
        else:
            self.getTangent = self.__computeTangent
    
    def setFluctuationUpdateFlag(self, flag):
        self.isFluctuationUpdated = flag
        
    def declareAuxVariables(self):
    
        self.dy = df.Measure('dx', self.mesh)
        self.vol = df.assemble(df.Constant(1.0)*self.dy)
        self.y = df.SpatialCoordinate(self.mesh)
        self.Gmacro = df.Constant((0.,0.,0.,0.,0.,0.,0.,0.,0.))  # just placeholder
        self.Gmacro_kl = df.Constant((0.,0.,0.,0.,0.,0.,0.,0.,0.))  # just placeholder
        
        self.duh = df.TrialFunction(self.Uh)            # Incremental displacement
        self.vh  = df.TestFunction(self.Uh)             # Test function
        self.uh  = df.Function(self.Uh)                 # Displacement from previous iteration
        self.ukl = df.Function(self.Uh)
        
        self.stresshom = np.zeros(self.nstrain)
        self.tangenthom = np.zeros((self.nstrain,self.nstrain))
        
    def setMicroproblem(self):
        
        dy, Gmacro = self.dy, self.Gmacro
        uh, vh, duh = self.uh, self.vh, self.duh
        
        Fmu = conv.Id_unsym_df + Gmacro + conv.grad_unsym(uh)
        Fmu_var = df.variable(Fmu)
        F = conv.unsym2tensor(Fmu_var)
        
        self.PKmu, self.Amu = ft.get_stress_tang_from_psi(self.psi_mu, F, Fmu_var) 
        
        self.Res = df.inner(self.PKmu , conv.grad_unsym(vh))*dy 
        self.Jac = df.inner(df.dot( self.Amu, conv.grad_unsym(duh)), conv.grad_unsym(vh))*dy
        
        self.microproblem = df.NonlinearVariationalProblem(self.Res, uh, self.bcD, self.Jac)
        self.microsolver = df.NonlinearVariationalSolver(self.microproblem)
        self.microsolver.parameters.update(solver_parameters)
    
    
    def setCanonicalproblem(self):
        dy, vh = self.dy, self.vh
                 
        # negative because 
        self.RHS_can = -df.inner(df.dot( self.Amu, self.Gmacro_kl), conv.grad_unsym(vh))*dy 
        
        self.Kcan = df.PETScMatrix()
        self.bcan = df.PETScVector()
        self.solver_can = df.PETScLUSolver()
        
    
    def __homogeniseTangent(self):
        
        # print("index" , GmacroMacro)
    
        dy, vol, Gmacro_kl, Amu, ukl = self.dy, self.vol, self.Gmacro_kl, self.Amu, self.ukl
        
        df.assemble(self.Jac, tensor = self.Kcan)
        self.bcD.apply(self.Kcan)
                        
        self.tangenthom.fill(0.0)
        
        unit_vec = np.zeros(self.nstrain)
        
        for i in range(self.nstrain):
            
            unit_vec[i] = 1.0
            Gmacro_kl.assign(df.Constant(unit_vec))
            
            df.assemble(self.RHS_can, tensor = self.bcan)
            self.bcD.apply(self.bcan)
    
            self.solver_can.solve(self.Kcan, ukl.vector(), self.bcan)
        
            self.tangenthom[i,:] += ft.Integral(df.dot(Amu, Gmacro_kl +  ft.grad_unsym(ukl)) , dy, ((self.nstrain,)))/vol
            
            unit_vec[i] = 0.0
        
        
        type(self).countComputeCanonicalProblem = type(self).countComputeCanonicalProblem + 1
        
        
    def __homogeniseStress(self):
        self.stresshom = ft.Integral(self.PKmu, self.dy, (self.nstrain,))/self.vol
        
    def __computeFluctuations(self, Gmacro):
        print("hello")
        self.restart_initial_guess()
        self.Gmacro.assign(df.Constant(Gmacro))
        self.microsolver.solve()
        self.setFluctuationUpdateFlag(True)
        
        type(self).countComputeFluctuations = type(self).countComputeFluctuations + 1

        
    def __computeStress(self, Gmacro):
        if(not self.isFluctuationUpdated):
            self.__computeFluctuations(Gmacro)
    
        self.__homogeniseStress()
        self.setStressUpdateFlag(True)
    
        return self.__returnStress(Gmacro)
        
    
    def __computeTangent(self, Gmacro):
        
        if(not self.isFluctuationUpdated):
            self.__computeFluctuations(Gmacro)
            
        self.__homogeniseTangent()
        self.setTangentUpdateFlag(True)            
        
        return self.__returnTangent(Gmacro)
        
