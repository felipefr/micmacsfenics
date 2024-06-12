#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:24:12 2022

@author: felipe
"""

import sys
sys.path.append("/home/felipe/sources/fetricks")
sys.path.append("/home/felipe/sources/micmacsfenics")
import numpy as np
import dolfin as df
from fetricks import symgrad, tensor2mandel
import micmacsfenics as mm
from timeit import default_timer as timer

df.parameters["form_compiler"]["representation"] = 'uflacs'
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3"

import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)


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

    str_fac = 'A*exp( -a*((x[0] - x0)*(x[0] - x0) + (x[1] - y0)*(x[1] - y0)) )'

    for xi, yi, ri in ellipseData[:, 0:3]:
        fac = fac + df.Expression(str_fac, A=contrast - 1.0,
                                  a=- np.log(radiusThreshold)/ri**2,
                                  x0=xi, y0=yi, degree=2)

    return fac


Lx = 2.0
Ly = 0.5

if(len(sys.argv)>2):
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
else:
    Nx = 10
    Ny = 5
    
facAvg = 1.0  # roughly chosen to approx single scale to mulsticale results
lamb = facAvg*1.0
mu = facAvg*0.5
alpha = 50.0
ty = -0.1

mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly),
                        Nx, Ny, "right/left")

start = timer()


deg_u = 1
deg_stress = 0
deg_tan = 0

Uh = df.VectorFunctionSpace(mesh, "CG", deg_u)
We = df.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=3, quad_scheme='default')
W = df.FunctionSpace(mesh, We)
W0e = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W0 = df.FunctionSpace(mesh, W0e)

Wten_e = df.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_tan, dim = 6, quad_scheme='default')
Wten = df.FunctionSpace(mesh, Wten_e)


metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
dxm = df.Measure('dx' , domain = mesh, metadata = metadata) 

leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx=Lx)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)

LoadBndFlag = 2

bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0)), boundary_markers, 1)

ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
traction = df.Constant((0.0, ty))

bc = [bcL]

def F_ext(v):
    return df.inner(traction, v)*ds(LoadBndFlag)

### Microscale 
nCells = mesh.num_cells()

name_meshmicro = 'meshmicro.xdmf'
NxMicro = NyMicro = 10
bndModel = 'lin' 

r0 = 0.3
r1 = 0.5

lamb_matrix = 1.0
mu_matrix = 0.5
LxMicro = LyMicro = 1.0
contrast = 1.0

meshMicro = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(LxMicro, LyMicro),
                             NxMicro, NyMicro, "right/left")

facs = [getFactorBalls(0) for i in range(nCells)]
np.random.seed(0)
params = [[fac_i*lamb_matrix, fac_i*mu_matrix, alpha] for fac_i in facs]
microModels = [mm.MicroConstitutiveModelNonlinear(meshMicro, pi, bndModel)
               for pi in params]


# ===================================================================================


hom = mm.MultiscaleModel(microModels, mesh, []) 
hom.createInternalVariables(W, Wten, dxm)

u = df.Function(Uh, name="Total displacement")
du = df.Function(Uh, name="Iteration correction")
v = df.TestFunction(Uh)
u_ = df.TrialFunction(Uh)

hom.update_alpha(tensor2mandel(symgrad(u)))

a_Newton = df.inner(tensor2mandel(symgrad(u_)), hom.tangent(tensor2mandel(symgrad(v))) )*dxm
res = -df.inner(tensor2mandel(symgrad(v)), hom.sig )*dxm + F_ext(v)

Nitermax, tol = 10, 1e-7  # parameters of the Newton-Raphson procedure

A, Res = df.assemble_system(a_Newton, res, bc)
nRes0 = Res.norm("l2")
nRes = nRes0
du.vector().set_local(np.zeros(Uh.dim()))
u.vector().set_local(np.zeros(Uh.dim()))

print(" Residual:", nRes)


niter = 0
while nRes/nRes0 > tol and niter < Nitermax:
    df.solve(A, du.vector(), Res)
    u.assign(u + du)
    hom.update_alpha(tensor2mandel(symgrad(u)))
    A, Res = df.assemble_system(a_Newton, res, bc)
    nRes = Res.norm("l2")
    print(" Residual:", nRes)
    niter += 1
    
end = timer()
print(end - start)


u.rename("uh", ".")
with df.XDMFFile("bar_multiscale_usingIntegrationPoints.xdmf") as f:
    f.write(u, 0.0)

