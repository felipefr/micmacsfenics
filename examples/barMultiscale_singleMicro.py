#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Availabel in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com,
felipe.figueredorocha@epfl.ch

Bar problem given a Multiscale constitutive law:
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left
and traction on the right.
The constitutive law is given implicitly by solving a micro problem in each
gauss point of micro-scale. (only on microstructure in the domain)s
We can choose the kinematically constrained model to the micro problem:
Linear, Periodic or Minimally Restricted
Entry to constitutive law: Mesh (micro), Lamé parameters (variable in micro),
Kinematical Model
"""

import sys
import dolfin as df
import numpy as np
sys.path.insert(0, '../utils/')
import multiscaleModels as mscm
from fenicsUtils import symgrad_voigt

r0 = 0.3
r1 = 0.5
Lx = 2.0
Ly = 0.5
Nx = 10
Ny = 3

lamb_matrix = 1.0
mu_matrix = 0.5
NxMicro = NyMicro = 100
LxMicro = LyMicro = 1.0
contrast = 10.0

ty = -0.01

# defining the micro model
Nballs = 4
ellipseData = np.zeros((Nballs, 3))
xlin = np.linspace(0.0 + 0.5/np.sqrt(Nballs), 1.0 - 0.5/np.sqrt(Nballs),
                   int(np.sqrt(Nballs)))
grid = np.meshgrid(xlin, xlin)
ellipseData[:, 0] = grid[0].flatten()
ellipseData[:, 1] = grid[1].flatten()
np.random.seed(1)
ellipseData[:, 2] = r0 + np.random.rand(Nballs)*(r1 - r0)

fac = df.Expression('1.0', degree=2)  # ground substance=
radiusThreshold = 0.01

str_fac = 'A*exp(-a*((x[0] - x0)*(x[0] - x0) + (x[1] - y0)*(x[1] - y0) ))'

for xi, yi, ri in ellipseData[:, 0:3]:
    fac = fac + df.Expression(str_fac, A=contrast - 1.0,
                              a=- np.log(radiusThreshold)/ri**2,
                              x0=xi, y0=yi, degree=2)

lamb = fac*lamb_matrix
mu = fac*mu_matrix

meshMicro = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(LxMicro, LyMicro),
                             NxMicro, NyMicro, "right/left")

microModel = mscm.MicroConstitutiveModel(meshMicro, [lamb, mu], 'per')
def sigma_voigt(w): return microModel.solveStress(w)


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

# Define variational problem
uh = df.TrialFunction(Uh)
vh = df.TestFunction(Uh)
a = df.inner(sigma_voigt(uh), symgrad_voigt(vh))*df.dx
b = df.inner(traction, vh)*ds(2)

# Compute solution
uh = df.Function(Uh)
df.solve(a == b, uh, bcL)

# Save solution in VTK format
fileResults = df.XDMFFile("barMultiscale_singleMicro.xdmf")
fileResults.write(uh)
