#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Availabel in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com,
felipe.figueredorocha@epfl.ch

Bar problem given a Macroscopic constitutive law:
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and traction on the
right. We use an isotropic linear material, given two lamé parameters.
"""
import sys
import dolfin as df
import numpy as np
from ufl import nabla_div
sys.path.insert(0, '../core/')
from fenicsUtils import symgrad

Lx = 1.0
Ly = 0.2
Nx = 400
Ny = 50

lamb = 1.0
mu = 0.5
ty = -1.0e-3

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
    return lamb*nabla_div(u)*df.Identity(2) + 2*mu*symgrad(u)


# Define variational problem
uh = df.TrialFunction(Uh)
vh = df.TestFunction(Uh)
a = df.inner(sigma(uh), df.grad(vh))*df.dx
b = df.inner(traction, vh)*ds(2)

# Compute solution
uh = df.Function(Uh)

# linear_solver ops: "superlu" or "mumps"
df.solve(a == b, uh, bcs=bcL, solver_parameters={"linear_solver": "superlu"})


print(uh.vector().get_local()[:].shape)
print(np.linalg.norm(uh.vector().get_local()[:]))

# Save solution in VTK format
fileResults = df.XDMFFile("barMacro.xdmf")
fileResults.write(uh)
