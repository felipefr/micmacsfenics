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

from micmacsfenics.core.fenicsUtils import symgrad, Integral, symgrad_voigt, macro_strain


from micmacsfenics.formulations.dirichlet_lagrange import FormulationDirichletLagrange
from micmacsfenics.formulations.linear import FormulationLinear
from micmacsfenics.formulations.periodic import FormulationPeriodic
from micmacsfenics.formulations.minimally_constrained import FormulationMinimallyConstrained

listMultiscaleModels = {'MR': FormulationMinimallyConstrained,
                        'per': FormulationPeriodic,
                        'lin': FormulationLinear,
                        'lag': FormulationDirichletLagrange, 
                        'dnn': FormulationDirichletLagrange} # add now


class MicroConstitutiveModel:

    def __init__(self, mesh, lame, model):
        def sigmaLaw(u):
            return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*symgrad(u)

        self.sigmaLaw = sigmaLaw

        self.mesh = mesh
        self.model = model
        self.coord_min = np.min(self.mesh.coordinates(), axis=0)
        self.coord_max = np.max(self.mesh.coordinates(), axis=0)

        # it should be modified before computing tangent (if needed)
        self.others = {
            'polyorder': 2,
            'x0': self.coord_min[0], 'x1': self.coord_max[0],
            'y0': self.coord_min[1], 'y1': self.coord_max[1]
            }

        self.multiscaleModel = listMultiscaleModels[model]
        self.x = df.SpatialCoordinate(self.mesh)
        self.ndim = 2
        self.nvoigt = int(self.ndim*(self.ndim + 1)/2)
        self.Chom_ = np.zeros((self.nvoigt, self.nvoigt))

        # in the first run should compute
        self.getTangent = self.computeTangent

    def computeTangent(self):

        dy = df.Measure('dx', self.mesh)
        vol = df.assemble(df.Constant(1.0)*dy)
        y = df.SpatialCoordinate(self.mesh)
        Eps = df.Constant(((0., 0.), (0., 0.)))  # just placeholder

        form = self.multiscaleModel(self.mesh, self.sigmaLaw, Eps, self.others)
        a, f, bcs, W = form()

        start = timer()
        A = mp.block_assemble(a)
        if(len(bcs) > 0):
            bcs.apply(A)

        # decompose just once (the faster for single process)
        solver = df.PETScLUSolver('superlu')
        sol = mp.BlockFunction(W)

        end = timer()
        print('time assembling system', end - start)

        for i in range(self.nvoigt):
            start = timer()
            
            if(i<2):
                Eps.assign(df.Constant(macro_strain(i)))
            else:
                Eps.assign(df.Constant(macro_strain(i)))
            
            F = mp.block_assemble(f)
            if(len(bcs) > 0):
                bcs.apply(F)

            solver.solve(A, sol.block_vector(), F)
            sol.block_vector().block_function().apply("to subfunctions")

            sig_mu = self.sigmaLaw(df.dot(Eps, y) + sol[0])
            sigma_hom = Integral(sig_mu, dy, (2, 2))/vol

            self.Chom_[:, i] = sigma_hom.flatten()[[0, 3, 1]]

            end = timer()
            print('time in solving system', end - start)

        # from the second run onwards, just returns
        self.getTangent = self.getTangent_

        return self.Chom_

    def getTangent_(self):
        return self.Chom_

    def solveStress(self, u):
        return df.dot(df.Constant(self.getTangent()), symgrad_voigt(u))
