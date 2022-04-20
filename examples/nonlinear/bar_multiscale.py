#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:24:12 2022

@author: felipe
"""

import sys
import dolfin as df
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '../../core/')
sys.path.insert(0, '../../materials/')


from fenicsUtils import symgrad, tensor2mandel,  mandel2tensor, tr_mandel, Id_mandel_df, Id_mandel_np
from hyperlastic_model import hyperlasticityModel
from homogenised_model import homogenisedModel, homogenisedHyperlasticityModel

from timeit import default_timer as timer

from functools import partial 

df.parameters["form_compiler"]["representation"] = 'uflacs'
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3"

import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

Lx = 2.0
Ly = 0.5

if(len(sys.argv)>2):
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
else:
    Nx = 6
    Ny = 3
    
facAvg = 4.0  # roughly chosen to approx single scale to mulsticale results
lamb = facAvg*1.0
mu = facAvg*0.5
alpha = 1.0
ty = -0.01

model = hyperlasticityModel({'lamb' : lamb, 'mu': mu, 'alpha': alpha})
model = hyperlasticityModel({'lamb' : lamb, 'mu': mu, 'alpha': alpha})


mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly),
                        Nx, Ny, "right/left")


start = timer()

deg_u = 1
deg_stress = 0
Uh = df.VectorFunctionSpace(mesh, "CG", deg_u)
We = df.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=3, quad_scheme='default')
W = df.FunctionSpace(mesh, We)
W0e = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W0 = df.FunctionSpace(mesh, W0e)

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

model.createInternalVariables(W, W0, dxm)

hom = homogenisedModel(model.eps, model.stress_np, model.tangent_np)

u = df.Function(Uh, name="Total displacement")
du = df.Function(Uh, name="Iteration correction")
v = df.TestFunction(Uh)
u_ = df.TrialFunction(Uh)

# sig_expr = sigma_law_(symgradexpr) 

a_Newton = df.inner(tensor2mandel(symgrad(u_)), df.dot(hom.tangent, tensor2mandel(symgrad(v))) )*dxm
res = -df.inner(tensor2mandel(symgrad(v)), hom.stress )*dxm + F_ext(v)

Nitermax, tol = 10, 1e-7  # parameters of the Newton-Raphson procedure

A, Res = df.assemble_system(a_Newton, res, bc)
nRes0 = Res.norm("l2")
nRes = nRes0
du.vector().set_local(np.zeros(Uh.dim()))
u.vector().set_local(np.zeros(Uh.dim()))

print(" Residual:", nRes)


niter = 0
while nRes/nRes0 > tol and niter < Nitermax:
    df.solve(A, du.vector(), Res, "superlu")
    u.assign(u + du)
    model.update_alpha(tensor2mandel(symgrad(u)))
    A, Res = df.assemble_system(a_Newton, res, bc)
    nRes = Res.norm("l2")
    print(" Residual:", nRes)
    niter += 1
    
end = timer()
print(end - start)


u.rename("uh", ".")
with df.XDMFFile("bar_multiscale.xdmf") as f:
    f.write(u, 0.0)

