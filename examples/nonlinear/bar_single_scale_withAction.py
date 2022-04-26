#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:04:28 2022

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
sys.path.insert(0, '../../core/')
from micmacsfenics.core.fenicsUtils import symgrad
from functools import partial 
from timeit import default_timer as timer

solver_parameters = {"nonlinear_solver": "newton",
                     "newton_solver": {"maximum_iterations": 20,
                                       "report": True,
                                       "error_on_nonconvergence": True}}

resultFolder = './results/'

def getSigma(u, param):
    lamb, mu, alpha = param
    
    e = symgrad(u)
    tr_e = df.tr(e)
    e2 = df.inner(e,e)
    
    return (lamb*(1 + alpha*tr_e**2)*tr_e*df.Identity(2) + 2*mu*(1 + alpha*e2)*e)


Lx = 2.0
Ly = 0.5

if(len(sys.argv)>2):
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
else:
    Nx = 10
    Ny = 3
    
facAvg = 1.0  # roughly chosen to approx single scale to mulsticale results
lamb = facAvg*1.0
mu = facAvg*0.5
alpha = 200.0
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


start = timer()
param = [lamb, mu, alpha]
sigma_law = partial(getSigma, param = param)

# Define variational problem
uh = df.TrialFunction(Uh)            # Incremental displacement
vh  = df.TestFunction(Uh)             # Test function
uh_  = df.Function(Uh)                 # Displacement from previous iteration
duh = df.TrialFunction(Uh)            # Incremental displacement

F = df.inner( sigma_law(uh), symgrad(vh))*df.dx - df.inner(traction, vh)*ds(2)

F_action = df.action(F, uh_)
 
# F = df.derivative(Pi, uh, vh)
J = df.derivative(F_action, uh_, duh)

problem = df.NonlinearVariationalProblem(F_action, uh_, bcL, J)
solver = df.NonlinearVariationalSolver(problem)
solver.parameters.update(solver_parameters)
solver.solve()

# Save solution in VTK format
# fileResults = df.XDMFFile(resultFolder + "bar_single_scale.xdmf")
# fileResults.write_checkpoint(uh, 'u', 0)

uh_.rename("uh", ".")
with df.XDMFFile("bar_single_scale.xdmf") as f:
    f.write(uh_, 0.0)


end = timer()
print(end - start)

y = df.interpolate(df.SpatialCoordinate(mesh), Uh)

J_action = df.action(J, symgrad(y))

b = df.assemble(J_action)