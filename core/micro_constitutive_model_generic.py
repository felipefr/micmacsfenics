#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 20:23:09 2023

@author: ffiguere
"""

import sys
import numpy as np
import dolfin as df
from timeit import default_timer as timer
from functools import partial 
import fetricks as ft


solver_parameters = {"nonlinear_solver": "newton",
                     "newton_solver": {"maximum_iterations": 20,
                                       "report": False,
                                       "error_on_nonconvergence": True}}

class MicroConstitutiveModelGeneric: # TODO derive it again from a base class

    # Counter of calls 
    countComputeFluctuations = 0
    countComputeCanonicalProblem = 0
    countTangentCalls = 0
    countStressCalls = 0
   
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
        # return np.concatenate((self.getStress(e), symflatten(self.getTangent(e))))
        return self.getStress(e), ft.sym_flatten_6x6_np(self.getTangent(e))

    def getStressTangent_force(self, e):
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
        
        psi_mu = partial(self.param["psi"], param = self.param)
        self.sigmu, self.Cmu = ft.get_stress_tang_from_psi(psi_mu, eps_var, eps_var) 
        
        self.Res = df.inner(self.sigmu , ft.symgrad_mandel(vh))*dy 
        self.Jac = df.inner(df.dot( self.Cmu, ft.symgrad_mandel(duh)), ft.symgrad_mandel(vh))*dy
        
        self.microproblem = df.NonlinearVariationalProblem(self.Res, uh, self.bcD, self.Jac)
        self.microsolver = df.NonlinearVariationalSolver(self.microproblem)
        self.microsolver.parameters.update(solver_parameters)
    
    
    def setCanonicalproblem(self):
        dy, vh = self.dy, self.vh
                 
        # negative because 
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
        
        
        type(self).countComputeCanonicalProblem = type(self).countComputeCanonicalProblem + 1
        
        
    def __homogeniseStress(self):
        self.stresshom = ft.Integral(self.sigmu, self.dy, (self.nvoigt,))/self.vol
        
    def __computeFluctuations(self, e):
        self.Eps.assign(df.Constant(e))
        self.microsolver.solve()
        self.setFluctuationUpdateFlag(True)
        
        type(self).countComputeFluctuations = type(self).countComputeFluctuations + 1

        
    def __computeStress(self, e):
        if(not self.isFluctuationUpdated):
            self.__computeFluctuations(e)
    
        self.__homogeniseStress()
        self.setStressUpdateFlag(True)
    
        return self.__returnStress(e)
        
    
    def __computeTangent(self, e):
        
        if(not self.isFluctuationUpdated):
            self.__computeFluctuations(e)
            
        self.__homogeniseTangent()
        self.setTangentUpdateFlag(True)            
        
        return self.__returnTangent(e)
        
