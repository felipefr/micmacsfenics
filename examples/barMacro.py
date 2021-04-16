#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Availabel in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com, felipe.figueredorocha@epfl.ch

Bar problem given a Macroscopic constitutive law:
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and traction on the right.
We use an isotropic linear material, given two lamé parameters.  
"""
import sys, os
from dolfin import *
import numpy as np
sys.path.insert(0,'../utils/')
import matplotlib.pyplot as plt
from ufl import nabla_div
from fenicsUtils import symgrad

Lx = 1.0
Ly = 0.2
Nx = 400
Ny = 50

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
    return lamb*nabla_div(u)*Identity(2) + 2*mu*symgrad(u)

# Define variational problem
uh = TrialFunction(Uh) 
vh = TestFunction(Uh)
a = inner(sigma(uh), grad(vh))*dx
b = inner(traction,vh)*ds(2)

# Compute solution
uh = Function(Uh)
solve(a == b, uh, bcs = bcL, solver_parameters={"linear_solver": "superlu"}) # normally the best for single process
# solve(a == b, uh, bcs = bcL, solver_parameters={"linear_solver": "mumps"}) # best for distributed 

print(uh.vector().get_local()[:].shape)
print(np.linalg.norm(uh.vector().get_local()[:]))

# Save solution in VTK format
fileResults = XDMFFile("barMacro.xdmf")
fileResults.write(uh)
