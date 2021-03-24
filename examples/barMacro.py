#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Availabel in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com, felipe.figueredorocha@epfl.ch

solves a bar problem
"""

from dolfin import *
import matplotlib.pyplot as plt
from ufl import nabla_div

Lx = 1.0
Ly = 0.2
Nx = 10
Ny = 3

lamb = 1.0
mu = 0.5
ty = -1.0e-3

# Create mesh and define function space
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny, "right/left")
Uh = VectorFunctionSpace(mesh, "Lagrange", 1)

leftBnd = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx = Lx)

boundary_markers = MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)

# Define boundary condition
bcL = DirichletBC(Uh, Constant((0.0,0.0)), boundary_markers, 1) # leftBnd instead is possible

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
traction = Constant((0.0,ty ))

def sigma(u):
    return lamb*nabla_div(u)*Identity(2) + 0.5*mu*sym(nabla_grad(u))

# Define variational problem
uh = TrialFunction(Uh) 
vh = TestFunction(Uh)
a = inner(sigma(uh), grad(vh))*dx
b = inner(traction,vh)*ds(2)

# Compute solution
uh = Function(Uh)
solve(a == b, uh, bcL)

# Save solution in VTK format
fileResults = XDMFFile("barMacro.xdmf")
fileResults.write(uh)
