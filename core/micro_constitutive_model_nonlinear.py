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
from ufl import nabla_div
from functools import partial 

from micmacsfenics.core.fenicsUtils import symgrad, Integral, symgrad_voigt, macro_strain_mandel, tensor2mandel_np, mandel2tensor_np
from micmacsfenics.formulations.dirichlet_lagrange import FormulationDirichletLagrange
from micmacsfenics.formulations.linear import FormulationLinear
from micmacsfenics.formulations.periodic import FormulationPeriodic
from micmacsfenics.formulations.minimally_constrained import FormulationMinimallyConstrained

from micmacsfenics.core.micro_constitutive_model import MicroConstitutiveModel


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

        self.psiLaw = partial(getPsi, param = param)
        self.sigmaLaw = partial(getSigma, param = param)
        
        self.onBoundary = df.CompiledSubDomain('on_boundary')
        self.Uh = df.VectorFunctionSpace(self.mesh, "CG", 1)     
        self.bcD = df.DirichletBC(self.Uh, df.Constant((0, 0)), self.onBoundary)   
        
    def computeTangent(self, e):
        
        # print("index" , epsMacro)
        
        dy = df.Measure('dx', self.mesh)
        vol = df.assemble(df.Constant(1.0)*dy)
        y = df.SpatialCoordinate(self.mesh)
        Eps = df.Constant(((0., 0.), (0., 0.)))  # just placeholder
        
        duh = df.TrialFunction(self.Uh)            # Incremental displacement
        vh  = df.TestFunction(self.Uh)             # Test function
        uh  = df.Function(self.Uh)                 # Displacement from previous iteration

        Pi = self.psiLaw(df.dot(Eps,y) + uh)*dy
         
        F = df.derivative(Pi, uh, vh)
        J = df.derivative(F, uh, duh)

        problem = df.NonlinearVariationalProblem(F, uh, self.bcD, J)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters.update(solver_parameters)
        
        for i in range(self.nvoigt):
            start = timer()
        
            Eps.assign(df.Constant(macro_strain_mandel(i)))
        
            solver.solve()
            
            sig_mu = self.sigmaLaw(df.dot(Eps, y) + uh)
            sigma_hom = Integral(sig_mu, dy, (2, 2))/vol

            self.Chom_[:, i] = tensor2mandel_np(sigma_hom)

            end = timer()
            print('time in solving system', end - start)


        # print(self.Chom_)
        # input()
        # from the second run onwards, just returns
        self.getTangent = self.getTangent_

        # print(self.Chom_)
        # input()
        return self.Chom_

    def getTangent_(self, e):
        return self.Chom_

    def getStress(self, e):
        
        dy = df.Measure('dx', self.mesh)
        vol = df.assemble(df.Constant(1.0)*dy)
        y = df.SpatialCoordinate(self.mesh)
        Eps = df.Constant(((0., 0.), (0., 0.)))  # just placeholder
        
        duh = df.TrialFunction(self.Uh)            # Incremental displacement
        vh  = df.TestFunction(self.Uh)             # Test function
        uh  = df.Function(self.Uh)                 # Displacement from previous iteration

        Pi = self.psiLaw(df.dot(Eps,y) + uh)*dy
         
        F = df.derivative(Pi, uh, vh)
        J = df.derivative(F, uh, duh)
        
        Eps.assign(df.Constant(mandel2tensor_np(e)))  # just placeholder
        
        problem = df.NonlinearVariationalProblem(F, uh, self.bcD, J)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters.update(solver_parameters)
        
        start = timer()
    
        solver.solve()
        
        sig_mu = self.sigmaLaw(df.dot(Eps, y) + uh)
        sigma_hom = Integral(sig_mu, dy, (2, 2))/vol
        
        end = timer()
        print('time in solving system', end - start)

        return tensor2mandel_np(sigma_hom)
        
        
