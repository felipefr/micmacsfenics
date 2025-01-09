#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2021-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

"""
Description:
Bar problem given a Multiscale constitutive law (random microstructures):
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and
traction on the right. The constitutive law is given implicitly by solving a
micro problem in each gauss point of micro-scale (one per element).
We can choose the kinematically constrained model to the
micro problem: Linear, Periodic or Minimally Restricted. Entry to constitutive
law: Mesh (micro), Lamé parameters (variable in micro), Kinematical Model
PS: This is a parallel implementation: random seed is chosen to match the
global cell index. Results are exactly the same as
bar_multiscale_standalone.py. Each local problem is solved in one process,
only the Macro-scale mesh is partitioned.
"""

import sys
import dolfin as df
import numpy as np
from mpi4py import MPI
import micmacsfenics as mm
import fetricks as ft
sys.path.insert(0, '../core/')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_ranks = comm.Get_size()
self_comm = MPI.COMM_SELF


class myChom(df.UserExpression):
    def __init__(self, microModels,  **kwargs):
        self.microModels = microModels
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        values[:] = self.microModels[cell.index].getTangent().flatten()

    def value_shape(self):
        return (3, 3,)


def getFactorBalls(seed=1):
    Nballs = 4
    ellipseData = np.zeros((Nballs, 3))
    xlin = np.linspace(0.0 + 0.5/np.sqrt(Nballs), 1.0 - 0.5/np.sqrt(Nballs),
                       int(np.sqrt(Nballs)))

    grid = np.meshgrid(xlin, xlin)
    ellipseData[:, 0] = grid[0].flatten()
    ellipseData[:, 1] = grid[1].flatten()
    np.random.seed(seed)
    ellipseData[:, 2] = r0 + np.random.rand(Nballs)*(r1 - r0)

    fac = df.Expression('1.0', degree=2)  # ground substance
    radiusThreshold = 0.01

    str_fac = 'A*exp(-a*((x[0] - x0)*(x[0] - x0) + (x[1] - y0)*(x[1] - y0) ))'

    for xi, yi, ri in ellipseData[:, 0:3]:
        fac = fac + df.Expression(str_fac, A=contrast - 1.0,
                                  a=- np.log(radiusThreshold)/ri**2,
                                  x0=xi, y0=yi, degree=2)

    return fac




if(len(sys.argv)>4):
    Lx = float(sys.argv[1])
    Ly = float(sys.argv[2])
    Nx = int(sys.argv[3])
    Ny = int(sys.argv[4])
    NxMicro = NyMicro = int(sys.argv[5]) #per, lin, 'MR', 'lag' or 'dnn' (not tested)
    bndModel = sys.argv[6]

else:    
    Lx = 2.0
    Ly = 0.5
    Nx = 40
    Ny = 10
    NxMicro = NyMicro = 10
    bndModel = 'per'    

r0 = 0.3
r1 = 0.5
lamb_matrix = 1.0
mu_matrix = 0.5

LxMicro = LyMicro = 1.0
contrast = 10.0

ty = -0.01

# Create mesh and define function space
mesh = df.RectangleMesh(comm, df.Point(0.0, 0.0), df.Point(Lx, Ly),
                        Nx, Ny, "right/left")
Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1)

leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx=Lx)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)


# defining the micro model
meshMicro = df.RectangleMesh(self_comm, df.Point(0.0, 0.0),
                             df.Point(LxMicro, LyMicro), NxMicro, NyMicro,
                             "right/left")

# random structure ruled by the global cell index
facs = [getFactorBalls(c.global_index()) for c in df.cells(mesh)]
params = [[fac_i*lamb_matrix, fac_i*mu_matrix] for fac_i in facs]
microModels = [mm.MicroConstitutiveModel(meshMicro, pi, bndModel)
               for pi in params]

Chom = myChom(microModels, degree=0)

# Define boundary condition
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0)), boundary_markers, 1)

ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
traction = df.Constant((0.0, ty))

# Define variational problem
uh = df.TrialFunction(Uh)
vh = df.TestFunction(Uh)
a = df.inner(df.dot(Chom, ft.symgrad_voigt(uh)), ft.symgrad_voigt(vh))*df.dx
b = df.inner(traction, vh)*ds(2)

# Compute solution
uh = df.Function(Uh)
df.solve(a == b, uh, bcL)

fileResults = df.XDMFFile(comm, "bar_multiscale_parallel_%s_np%d.xdmf" %
                          (bndModel, num_ranks))
fileResults.write_checkpoint(uh, 'u', 0)
