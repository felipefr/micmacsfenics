#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:33:32 2022

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
from ufl import nabla_div
from functools import partial 


from micmacsfenics.core.fenicsUtils import symgrad, Integral, symgrad_voigt, macro_strain


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
    lamb, mu = param
    
    e = symgrad(u)
    tr_e = df.tr(e)
    e2 = df.inner(e,e)
    
    return (0.5*lamb*(tr_e**2) + mu*(e2))

def getSigma(u, param): # linear elastic one
    lamb, mu = param
    
    e = symgrad(u)
    tr_e = df.tr(e)
    
    return (lamb*tr_e*df.Identity(2) + 2.0*mu*e)

class MicroConstitutiveModelEps(MicroConstitutiveModel):

    def __init__(self, mesh, lame, model):
        super().__init__(mesh, lame, model)

        self.psiLaw = partial(getPsi, param = lame)
        self.sigmaLaw = partial(getSigma, param = lame)
        
        self.onBoundary = df.CompiledSubDomain('on_boundary')
        self.Uh = df.VectorFunctionSpace(self.mesh, "CG", self.others['polyorder'])     
        self.bcD = df.DirichletBC(self.Uh, df.Constant((0, 0)), self.onBoundary)   
        
    def computeTangent(self, epsMacro):
        
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
        
            Eps.assign(df.Constant(macro_strain(i)))
        
            solver.solve()
            
            sig_mu = self.sigmaLaw(df.dot(Eps, y) + uh)
            sigma_hom = Integral(sig_mu, dy, (2, 2))/vol

            self.Chom_[:, i] = sigma_hom.flatten()[[0, 3, 1]]

            end = timer()
            print('time in solving system', end - start)
            
        # from the second run onwards, just returns
        # self.getTangent = self.getTangent_

        # print(self.Chom_)
        # input()
        return self.Chom_

    def getTangent_(self):
        return self.Chom_

    def solveStress(self, u):
        return df.dot(df.Constant(self.getTangent()), symgrad_voigt(u))
