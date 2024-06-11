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
Bar problem given a constitutive law (single-scale):
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and traction on the
right. We use an isotropic linear material, given two lamé parameters.
"""
import sys
sys.path.append("/home/felipe/sources/fetricks")
import dolfin as df
import numpy as np
from ufl import nabla_div
import fetricks as ft

resultFolder = '../results/'


if(len(sys.argv)>4):
    Lx = float(sys.argv[1])
    Ly = float(sys.argv[2])
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
else:    
    Lx = 2.0
    Ly = 0.5
    Nx = 40
    Ny = 10
    
facAvg = 4.0  # roughly chosen to approx single scale to mulsticale results
lamb = facAvg*1.0
mu = facAvg*0.5
ty = -0.01

# Create mesh and define function space
mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly),
                        Nx, Ny, "right/left")

Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1)

leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx=Lx)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)

# Define boundary condition
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0)), boundary_markers, 1)

ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
traction = df.Constant((0.0, ty))


def sigma(u):
    return lamb*nabla_div(u)*df.Identity(2) + 2*mu*ft.symgrad(u)


# Define variational problem
uh = df.TrialFunction(Uh)
vh = df.TestFunction(Uh)
a = df.inner(sigma(uh), df.grad(vh))*df.dx
b = df.inner(traction, vh)*ds(2)

# Compute solution
uh = df.Function(Uh)

# linear_solver ops: "superlu" or "mumps"
df.solve(a == b, uh, bcs=bcL, solver_parameters={"linear_solver": "superlu"})

# Save solution in VTK format
# fileResults = df.XDMFFile(resultFolder + "bar_single_scale.xdmf")
# fileResults.write_checkpoint(uh, 'u', 0)


uh.rename("uh", ".")
with df.XDMFFile (resultFolder + "bar_single_scale.xdmf",) as f:
    f.write_checkpoint(uh, 'uh', 0.0)