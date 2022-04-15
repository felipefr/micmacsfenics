#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 21:30:58 2022

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Available in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com,
felipe.figueredorocha@epfl.ch

Bar problem given a constitutive law (single-scale):
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and traction on the
right. We use an isotropic linear material, given two lamé parameters.
"""
import sys
import dolfin as df
import numpy as np
from ufl import nabla_div
sys.path.insert(0, '../core/')
from fenicsUtils import symgrad_voigt, symgrad, voigt2strain


# Optimization options for the form compiler
df.parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
                "eliminate_zeros": True, \
                "precompute_basis_const": True, \
                "precompute_ip_const": True}

resultFolder = './results/'


Lx = 2.0
Ly = 0.5

if(len(sys.argv)>2):
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
else:
    Nx = 40
    Ny = 10
    

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = df.Constant(E/(2*(1 + nu))), df.Constant(E*nu/((1 + nu)*(1 - 2*nu)))

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

dx = df.Measure('dx', domain=mesh)

# Define functions
duh = df.TrialFunction(Uh)            # Incremental displacement
vh  = df.TestFunction(Uh)             # Test function
uh  = df.Function(Uh)                 # Displacement from previous iteration

# Kinematics
eps = symgrad_voigt(uh)
tr_eps = eps[0] + eps[1]
eps2  = eps[0]**2 + eps[1]**2 + 0.5*eps[2]**2

# Define variational problem
psi = 0.5*lmbda*tr_eps**2  + mu*eps2

eps_var = df.variable(eps)
sigma = df.diff(psi,eps_var)
Celas = df.diff(sigma, eps_var)

# Pi = psi*dx - df.inner(traction, uh)*ds

Res = -df.inner(sigma, symgrad_voigt(vh))*dx + df.inner(traction, vh)*ds 
Jac = df.inner(Celas*symgrad_voigt(duh), symgrad_voigt(vh))*dx

# duh = df.Function(uh)
duh = df.Function(Uh)

df.solve(Jac == Res, duh, bcL)

# # Save solution in VTK format
# fileResults = df.XDMFFile(resultFolder + "bar_withStress.xdmf")
# fileResults.write_checkpoint(uh, 'u', 0)
# fileResults.close()