#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Availabel in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com, felipe.figueredorocha@epfl.ch

solves a bar problem
"""

from dolfin import *
import matplotlib.pyplot as plt

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
Uh = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(Uh, u0, boundary)

# Define variational problem
uh = TrialFunction(Uh) 
vh = TestFunction(Uh)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(uh), grad(vh))*dx
b = f*vh*dx + g*vh*ds

# Compute solution
uh = Function(Uh)
solve(a == b, uh, bc)

# Plot solution
plot(uh)
plt.show()

# Save solution in VTK format
# file = File("poisson.pvd")
# file << uh