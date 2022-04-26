#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:52:43 2022

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:06:25 2022

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:52:32 2021

@author: felipefr
"""

import sys
import numpy as np
import multiphenics as mp
import dolfin as df
from timeit import default_timer as timer
from ufl import nabla_div, indices
from functools import partial 

from micmacsfenics.core.fenicsUtils import symgrad, Integral, symgrad_voigt, macro_strain_mandel, tensor2mandel_np, mandel2tensor_np
from micmacsfenics.formulations.dirichlet_lagrange import FormulationDirichletLagrange
from micmacsfenics.formulations.linear import FormulationLinear
from micmacsfenics.formulations.periodic import FormulationPeriodic
from micmacsfenics.formulations.minimally_constrained import FormulationMinimallyConstrained

from micmacsfenics.core.micro_constitutive_model import MicroConstitutiveModel

from fenicsUtils import (symgrad, tensor2mandel, tensor4th2mandel, mandel2tensor, tr_mandel, Id_mandel_df,
                        Id_mandel_np, LocalProjector)

solver_parameters = {"nonlinear_solver": "newton",
                     "newton_solver": {"maximum_iterations": 20,
                                       "report": False,
                                       "error_on_nonconvergence": True}}

def getPsi(u, param): # linear elastic one
    lamb, mu, alpha = param
    
    e = symgrad(u)
    tr_e = df.tr(e)
    e2 = df.inner(e,e)
    
    return (0.5*lamb*(1.0 + 0.5*alpha*(tr_e**2))*(tr_e**2) + mu*(1 + 0.5*alpha*e2)*(e2))

def getSigma(u, param): # linear elastic one
    lamb, mu, alpha = param
    
    e = symgrad(u)
    tr_e = df.tr(e)
    e2 = df.inner(e,e)
    
    return (lamb*(1.0 + alpha*(tr_e**2))*tr_e*df.Identity(2) + 2.0*mu*(1.0 + alpha*e2)*e)


class MicroConstitutiveModelNonlinear(MicroConstitutiveModel):

    def __init__(self, mesh, param, model):
        super().__init__(mesh, param, model)

        self.param = param
        self.psiLaw = partial(getPsi, param = param)
        self.sigmaLaw = partial(getSigma, param = param)
        
        self.onBoundary = df.CompiledSubDomain('on_boundary')
        self.Uh = df.VectorFunctionSpace(self.mesh, "CG", 1)     
        self.bcD = df.DirichletBC(self.Uh, df.Constant((0, 0)), self.onBoundary)   
        
        self.isUpdated = False
    
        self.declareAuxVariables()
        self.setMicroproblem()
        self.setCanonicalproblem()
    
    def declareAuxVariables(self):
    
        self.dy = df.Measure('dx', self.mesh)
        self.vol = df.assemble(df.Constant(1.0)*self.dy)
        self.y = df.SpatialCoordinate(self.mesh)
        self.Eps = df.Constant((0., 0., 0.))  # just placeholder
        self.Eps_kl = df.Constant((0., 0., 0.))  # just placeholder
        
        self.duh = df.TrialFunction(self.Uh)            # Incremental displacement
        self.vh  = df.TestFunction(self.Uh)             # Test function
        self.uh  = df.Function(self.Uh)                 # Displacement from previous iteration
    
        self.stresshom = np.zeros(self.nvoigt)
        self.tangenthom = np.zeros((self.nvoigt,self.nvoigt))
        
    def setMicroproblem(self):
        
        dy, y, Eps = self.dy, self.y, self.Eps
        uh, vh, duh = self.uh, self.vh, self.duh
    
        Pi = self.psiLaw(df.dot(mandel2tensor(Eps),y) + uh)*dy
         
        F = df.derivative(Pi, uh, vh)
        J = df.derivative(F, uh, duh)
        
        self.microproblem = df.NonlinearVariationalProblem(F, uh, self.bcD, J)
        self.microsolver = df.NonlinearVariationalSolver(self.microproblem)
        self.microsolver.parameters.update(solver_parameters)
    
    
    def setCanonicalproblem(self):
        
        dy, vol, y, Eps = self.dy, self.vol, self.y, self.Eps
        uh, vh, duh = self.uh, self.vh, self.duh
         
        eps = self.Eps  + tensor2mandel(symgrad(self.uh))             # Deformation gradient
        eps_var = df.variable(eps)
        
        tr_e = tr_mandel(eps_var)
        e2 = df.inner(eps_var, eps_var)

        lamb, mu, alpha = self.param
        
        psi_mu = (0.5*lamb*(1.0 + 0.5*alpha*(tr_e**2))*(tr_e**2) + mu*(1 + 0.5*alpha*e2)*(e2))
        
        sigmu = df.diff(psi_mu , eps_var)
        self.Cmu = df.diff(sigmu, eps_var)
        
        # Pi = psi*dx - df.inner(traction, uh)*ds
        
        self.LHS = df.inner(df.dot( self.Cmu, tensor2mandel(symgrad(duh))), tensor2mandel(symgrad(vh)))*dy
        
        self.RHS = -df.inner(df.dot( self.Cmu, self.Eps_kl), tensor2mandel(symgrad(vh)))*dy 
        
    
    def computeTangent(self, e):
        
        # print("index" , epsMacro)
    
        self.setCanonicalproblem()
        dy, vol, y, Eps, Eps_kl, Cmu = self.dy, self.vol, self.y, self.Eps, self.Eps_kl, self.Cmu
        uh, vh, duh = self.uh, self.vh, self.duh
        
        Eps.assign(df.Constant(e))
        
        A = df.assemble(self.LHS)
        self.bcD.apply(A)
                        
        solver = df.PETScLUSolver()
        
        self.Chom_ = Integral(Cmu, dy, (self.nvoigt,self.nvoigt)) # just Chom_bar
        
        ukl = df.Function(self.Uh)
        
        unit_vec = np.zeros(self.nvoigt)
        
        for i in range(self.nvoigt):
            start = timer()
            
            unit_vec[i] = 1.0
            self.Eps_kl.assign(df.Constant(unit_vec))
            
            b = df.assemble(self.RHS)
            self.bcD.apply(b)

            solver.solve(A, ukl.vector(), b)
            
            # Chom fluc
            self.Chom_[i,:] = self.Chom_[i,:] + Integral(df.dot(Cmu, tensor2mandel(symgrad(ukl))) , dy, ((self.nvoigt,)))
            
            unit_vec[i] = 0.0
        
        
        end = timer()
        print('time in solving system', end - start)

    

        # print(self.Chom_)
        # input()
        # from the second run onwards, just returns
        # self.getTangent = self.getTangent_

        # print(self.Chom_)
        # input()
        return self.Chom_

    def getTangent_(self, e):
        return self.Chom_

    def getStress(self, e):
        
        self.Eps.assign(df.Constant(e))  # just placeholder
        
        start = timer()
    
        self.microsolver.solve()
        
        sig_mu = self.sigmaLaw(df.dot(mandel2tensor(self.Eps), self.y) + self.uh)
        sigma_hom = Integral(sig_mu, self.dy, (2, 2))/self.vol
        
        self.stresshom[:] = tensor2mandel_np(sigma_hom)
        
        end = timer()
        print('time in solving system', end - start)

        return self.stresshom
        
        
