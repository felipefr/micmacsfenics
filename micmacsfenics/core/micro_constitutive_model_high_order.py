#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:02:22 2025

@author: frocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2025, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or <felipe.f.rocha@gmail.com>

"""

import sys
import numpy as np
import dolfin as df
from timeit import default_timer as timer
from functools import partial 
import fetricks as ft
import multiphenics as mp
from ufl import nabla_div
import ufl

from micmacsfenics.formulations.dirichlet_lagrange import FormulationDirichletLagrange
from micmacsfenics.formulations.linear import FormulationLinear
from micmacsfenics.formulations.periodic import FormulationPeriodic
from micmacsfenics.formulations.minimally_constrained import FormulationMinimallyConstrained
from micmacsfenics.formulations.minimally_constrained_high_order import FormulationMinimallyConstrainedHighOrder


solver_parameters = {"nonlinear_solver": "newton",
                     "newton_solver": {"maximum_iterations": 20,
                                       "report": False,
                                       "error_on_nonconvergence": True}}


listMultiscaleModels = {'MR': FormulationMinimallyConstrained,
                        'per': FormulationPeriodic,
                        'lin': FormulationLinear,
                        'lag': FormulationDirichletLagrange, 
                        'dnn': FormulationDirichletLagrange,
                        'MRHO': FormulationMinimallyConstrainedHighOrder}


class MicroConstitutiveModelHighOrder: # TODO derive it again from a base class

    # Counter of calls 
    countComputeFluctuations = 0
    countComputeCanonicalProblem = 0
    countTangentCalls = 0
    countStressCalls = 0
   
    def __init__(self, mesh, psi_mu, bnd_model = []):
        self.mesh = mesh
        self.ndim = 2
        self.nvoigt = int(self.ndim*(self.ndim + 1)/2)
        self.tensor_encoding = "mandel"        
        self.bnd_model = bnd_model
        
        self.psi_mu = psi_mu
        
        
        self.Uh = df.VectorFunctionSpace(self.mesh, "CG", 1)     
        
        
        # by default linear model "lin"
        if(len(bnd_model)<2): 
            onBoundary = df.CompiledSubDomain('on_boundary')
            self.bcD = [df.DirichletBC(self.Uh, df.Constant((0, 0)), onBoundary)]
        else:
            self.bcD = [df.DirichletBC(self.Uh, df.Constant((0, 0)), self.mesh.boundaries, i) for i in bnd_model[1] ]  
            
        self.isStressUpdated = False
        self.isTangentUpdated = False
        self.isFluctuationUpdated = False
        

        self.getStress = self.__computeStress
        self.getTangent = self.__computeTangent

        self.declareAuxVariables()
        self.setMicroproblem()
        # self.setCanonicalproblem()
    
    
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
        return self.getStress(e), ft.sym_flatten_3x3_np(self.getTangent(e)) # already in the mandel format

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
        self.Eps = df.Constant((0., 0., 0.))  # just placeholder
        self.Eps_kl = df.Constant((0., 0., 0.))  # just placeholder
        
        self.duh = df.TrialFunction(self.Uh)            # Incremental displacement
        self.vh  = df.TestFunction(self.Uh)             # Test function
        self.uh  = df.Function(self.Uh)                 # Displacement from previous iteration
        self.ukl = df.Function(self.Uh)
        
        self.stresshom = np.zeros(self.nvoigt)
        self.tangenthom = np.zeros((self.nvoigt,self.nvoigt))
        
        # below, it is useful to more general boundaries with multiphenics
        self.coord_min = np.min(self.mesh.coordinates(), axis=0)
        self.coord_max = np.max(self.mesh.coordinates(), axis=0)
        J = ft.Integral(df.outer(self.y,self.y), self.mesh.dx, shape=(2,2))
        Jinv = df.Constant(np.linalg.inv(J))
        self.others = {
            'polyorder': 1,
            'x0': self.coord_min[0], 'x1': self.coord_max[0],
            'y0': self.coord_min[1], 'y1': self.coord_max[1],
            'Jinv': Jinv
            }
        if(len(self.bnd_model)==2):
            self.others['external_bnd'] = self.bnd_model[1]

        
    def setMicroproblem(self):
        
        dy, Eps = self.dy, self.Eps
        uh, vh, duh = self.uh, self.vh, self.duh
        
        eps = Eps  + ft.symgrad_mandel(uh)            # Deformation gradient
        eps_var = df.variable(eps)
        
        self.sigmu, self.Cmu = ft.get_stress_tang_from_psi(self.psi_mu, eps_var, eps_var) 
        
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
        for bc in self.bcD:
            bc.apply(self.Acan)
                        
        self.tangenthom.fill(0.0)
        
        unit_vec = np.zeros(self.nvoigt)
        
        for i in range(self.nvoigt):
            
            unit_vec[i] = 1.0
            Eps_kl.assign(df.Constant(unit_vec))
            
            df.assemble(self.RHS_can, tensor = self.bcan)
            for bc in self.bcD:
                bc.apply(self.bcan)
    
            self.solver_can.solve(self.Acan, ukl.vector(), self.bcan)
        
            self.tangenthom[i,:] += ft.Integral(df.dot(Cmu, Eps_kl +  ft.symgrad_mandel(ukl)) , dy, ((self.nvoigt,)))/vol
            
            unit_vec[i] = 0.0
        
        
        type(self).countComputeCanonicalProblem = type(self).countComputeCanonicalProblem + 1
        
        
    def __homogeniseStress(self):
        self.stresshom = ft.Integral(self.sigmu, self.dy, (self.nvoigt,))/self.vol
        
    def __computeFluctuations(self, e):
        self.restart_initial_guess()
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

# This is the most mature implementation
    def compute_tangent_localisation_tensors(self):
        
        self.multiscaleModel = listMultiscaleModels[self.bnd_model[0]]
        form = self.multiscaleModel(self.mesh, self.others)
        W = form.W 
        sol = mp.BlockFunction(W)
        
        eps = self.Eps  + ft.symgrad_mandel(sol[0]) # micro-strain
        eps_var = df.variable(eps)
        self.sigmu, self.Cmu = ft.get_stress_tang_from_psi(self.psi_mu, eps_var, eps_var) 

        a, f, bcs, W = form(self.Cmu, self.Eps)       
        
        start = timer()
        A = mp.block_assemble(a)
        if(len(bcs) > 0):
            bcs.apply(A)

        # decompose just once (the faster for single process)
        solver = df.PETScLUSolver()
        
        end = timer()
        print('time assembling system', end - start)
        
        eye = ufl.Identity(self.nvoigt)
        S = eye # localization tensor
        for i_ in range(self.nvoigt):
            start = timer()            
            self.Eps.assign(df.Constant(eye[i_,:]))
            
            F = mp.block_assemble(f)
            if(len(bcs) > 0):
                bcs.apply(F)

            solver.solve(A, sol.block_vector(), F)
            sol.block_vector().block_function().apply("to subfunctions")
            
            S += ufl.outer(ft.symgrad_mandel(sol[0]), eye[i_,:])
            
        ij, kl, pq, rs = ufl.indices(4)
        self.tangenthom[:,:] = ft.Integral( df.as_tensor(S[pq,ij]*self.Cmu[pq,rs]*S[rs,kl] , (ij,kl)), 
                                         self.mesh.dx, (self.nvoigt, self.nvoigt))/self.vol
        end = timer()
        print('time in solving system', end - start)

        return self.tangenthom

# hyper tangent (high-order)
    def compute_hypertangent(self):
        
        nmandel3rd = 6
        H = df.Constant(np.zeros(nmandel3rd))
        H_full = ft.mandel2tensor3rd(H)
        
        self.multiscaleModel = listMultiscaleModels[self.bnd_model[0]]
        form = self.multiscaleModel(self.mesh, self.others)
        W = form.W 
        sol = mp.BlockFunction(W)
        
        eps = self.Eps  + ft.tensor2mandel(df.dot(H_full, self.y)) + ft.symgrad_mandel(sol[0]) # micro-strain
        eps_var = df.variable(eps)
        self.sigmu, self.Cmu = ft.get_stress_tang_from_psi(self.psi_mu, eps_var, eps_var) 

        a, f, bcs, W = form(self.Cmu, ft.tensor2mandel(df.dot(H_full, self.y)))       
        
        start = timer()
        A = mp.block_assemble(a)
        if(len(bcs) > 0):
            bcs.apply(A)

        # decompose just once (the faster for single process)
        solver = df.PETScLUSolver()
        
        end = timer()
        print('time assembling system', end - start)
        
        eye = ufl.Identity(nmandel3rd)
        # show be according to the Stilde below (not matching dimension)
        S = ufl.outer(ufl.Identity(self.nvoigt), self.y)   # localization tensor
        for i_ in range(nmandel3rd):
            start = timer()            
            H.assign(df.Constant(eye[i_,:]))
            
            F = mp.block_assemble(f)
            if(len(bcs) > 0):
                bcs.apply(F)

            solver.solve(A, sol.block_vector(), F)
            sol.block_vector().block_function().apply("to subfunctions")
            
            S += ufl.outer(ft.symgrad_mandel(sol[0]), eye[i_,:])
            
        ijk, lmn, pq, rs = ufl.indices(4)
        hypertanghom = ft.Integral( df.as_tensor(S[pq,ijk]*self.Cmu[pq,rs]*S[rs,lmn] , (ijk,lmn)), 
                                         self.mesh.dx, (nmandel3rd, nmandel3rd))/self.vol
        end = timer()
        print('time in solving system', end - start)

        return hypertanghom



    def compute_tangent_multiphenics(self):
        
        
        self.multiscaleModel = listMultiscaleModels[self.bnd_model[0]]
        
        lamb = df.Constant(432.099)
        mu = df.Constant(185.185)
        
        i, j, k, l = ufl.indices(4)
        delta = ufl.Identity(2)
        
        Is = df.as_tensor(0.5*(delta[i,k]*delta[j,l] + delta[i,l]*delta[j,k]), (i,j,k,l))
        Cmu = df.as_tensor(lamb*delta[i,j]*delta[k,l] + 2*mu*Is[i,j,k,l], (i,j,k,l))
        
        
        def sigmaLaw(u):
            return lamb*nabla_div(u)*df.Identity(2) + 2*mu*ft.symgrad(u)
        
        self.tangenthom_mp = np.zeros((self.nvoigt,self.nvoigt))
        
        self.sigmaLaw = sigmaLaw
        
        Eps = df.Constant(((0., 0.), (0., 0.)))  # just placeholder

        form = self.multiscaleModel(self.mesh, self.others)
        a, f, bcs, W = form( ft.tensor4th2mandel(Cmu), ft.tensor2mandel(Eps))

        start = timer()
        A = mp.block_assemble(a)
        if(len(bcs) > 0):
            bcs.apply(A)

        # decompose just once (the faster for single process)
        solver = df.PETScLUSolver()
        sol = mp.BlockFunction(W)

        end = timer()
        print('time assembling system', end - start)

        i, j = ufl.indices(2)
        
        umu = ufl.as_tensor( Eps[i,j]*self.y[j] + sol[0][i] , (i,))
        sig_mu = self.sigmaLaw(umu)

        for i_ in range(self.nvoigt):
            start = timer()
            
            Eps.assign(df.Constant(ft.macro_strain_mandel(i_)))
            
            F = mp.block_assemble(f)
            if(len(bcs) > 0):
                bcs.apply(F)

            solver.solve(A, sol.block_vector(), F)
            sol.block_vector().block_function().apply("to subfunctions")
            sigma_hom = ft.tensor2mandel(ft.Integral(sig_mu, self.mesh.dx, (2, 2))/self.vol)

            self.tangenthom_mp[:, i_] = sigma_hom[:]

            end = timer()
            print('time in solving system', end - start)


        return self.tangenthom_mp


    def compute_tangent_localisation_tensors_full_notation(self):
        
        
        self.multiscaleModel = listMultiscaleModels[self.bnd_model[0]]
        
        lamb = df.Constant(432.099)
        mu = df.Constant(185.185)
        
        i, j, k, l = ufl.indices(4)
        delta = ufl.Identity(2)
        
        Is = df.as_tensor(0.5*(delta[i,k]*delta[j,l] + delta[i,l]*delta[j,k]), (i,j,k,l))
        Cmu = df.as_tensor(lamb*delta[i,j]*delta[k,l] + 2*mu*Is[i,j,k,l], (i,j,k,l))
        
        Sbar = Is
        
        def sigmaLaw(u):
            return df.as_tensor(Cmu[i,j,k,l]*u[k].dx(l), (i,j)) 
        
        self.tangenthom_mp = np.zeros((self.nvoigt,self.nvoigt))
        
        self.sigmaLaw = sigmaLaw
        
        Eps = df.Constant(((0., 0.), (0., 0.)))  # just placeholder

        form = self.multiscaleModel(self.mesh, self.others)
        a, f, bcs, W = form( ft.tensor4th2mandel(Cmu), ft.tensor2mandel(Eps))

        start = timer()
        A = mp.block_assemble(a)
        if(len(bcs) > 0):
            bcs.apply(A)

        # decompose just once (the faster for single process)
        solver = df.PETScLUSolver()
        sol = mp.BlockFunction(W)

        end = timer()
        print('time assembling system', end - start)

        list_sol_can = []        

        eye = np.eye(2)
        for i_ in range(2):
            for j_ in range(2):
                list_sol_can.append(df.Function(self.Uh))
                start = timer()
                
                Eps.assign(df.Constant(np.outer(eye[i_,:],eye[j_,:])))
                
                F = mp.block_assemble(f)
                if(len(bcs) > 0):
                    bcs.apply(F)
    
                solver.solve(A, sol.block_vector(), F)
                sol.block_vector().block_function().apply("to subfunctions")
                
                list_sol_can[-1].assign(sol[0])
                
        expr = ( list_sol_can[0][i].dx(j)*delta[0,k]*delta[0,l]
             + list_sol_can[1][i].dx(j)*delta[0,k]*delta[1,l]
             + list_sol_can[2][i].dx(j)*delta[1,k]*delta[0,l]
             + list_sol_can[3][i].dx(j)*delta[1,k]*delta[1,l])
             
        S = Sbar + df.as_tensor( expr, (i,j,k,l))
        
        p,q,r,s = ufl.indices(4)
        Chom = ft.Integral( df.as_tensor(S[p,q,i,j]*Cmu[p,q,r,s]*S[r,s,k,l] , (i,j,k,l)), 
                           self.mesh.dx, (2, 2, 2, 2))/self.vol

        self.tangenthom_mp = ft.tensor4th2mandel_np(Chom)

        end = timer()
        print('time in solving system', end - start)

        return self.tangenthom_mp