#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:56:55 2024

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2021-2024, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

"""
Available in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, felipe.figueredo-rocha@u-pec.fr, f.rocha.felipe@gmail.com,


Bar problem given a constitutive law (single-scale):
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and traction on the
right. We use an isotropic linear material, given two lamé parameters.
"""


import os, sys
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from functools import partial 

sys.path.append("/home/felipe/sources/fetricksx")
sys.path.append("/home/felipe/sources/micmacsfenicsx/core")

import basix
import numpy as np
from dolfinx import fem, io
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from mpi4py import MPI

import fetricksx as ft
from micro_model import MicroModel
from micromacro import MicroMacro


class CustomQuadratureSpace:
    
    def __init__(self, mesh, dim, degree_quad = None):
        self.degree_quad = degree_quad
        self.mesh = mesh
        self.basix_cell = self.mesh.basix_cell()
        
        self.dxm = ufl.Measure("dx", domain=self.mesh, metadata={"quadrature_degree": self.degree_quad, "quadrature_scheme": "default"})
        self.W0e = basix.ufl.quadrature_element(self.basix_cell, degree=self.degree_quad, scheme = "default", value_shape= ())
        self.We = basix.ufl.quadrature_element(self.basix_cell, degree=self.degree_quad, scheme = "default", value_shape = (dim,))
        self.space = fem.functionspace(self.mesh, self.We)       
        self.scalar_space = fem.functionspace(self.mesh, self.W0e)
        basix_celltype = getattr(basix.CellType, self.mesh.topology.cell_type.name)
        points, weights = basix.make_quadrature(basix_celltype, self.degree_quad)
        self.eval_points = points
        self.weights = weights



def getMicroModel(mesh_micro_name= "../meshes/mesh_micro.xdmf"):
        
    # mesh_micro_name = 'meshes/mesh_micro.xdmf'
    lamb_ref = 432.099
    mu_ref = 185.185
    contrast = 50
    bndModel_micro = 'lin'
    vf = 0.124858 # volume fraction of inclusions vf*c_i + (1-vf)*c_m
    psi_mu = ft.psi_hookean_nonlinear_lame
    alpha_micro_m = 10000.0
    alpha_micro_i = 10000.0
    # alpha_micro_m = 0.0
    # alpha_micro_i = 0.0

    lamb_micro_m = lamb_ref/( 1-vf + contrast*vf)
    mu_micro_m = mu_ref/( 1-vf + contrast*vf)
    lamb_micro_i = contrast*lamb_micro_m
    mu_micro_i = contrast*mu_micro_m
    
    mesh_micro = ft.Mesh(mesh_micro_name)
    
    alpha_ = ft.create_piecewise_constant_field(mesh_micro, mesh_micro.markers, 
                                               {0: alpha_micro_i, 1: alpha_micro_m})
    lamb_ = ft.create_piecewise_constant_field(mesh_micro, mesh_micro.markers, 
                                               {0: alpha_micro_i, 1: alpha_micro_m})
    mu_ = ft.create_piecewise_constant_field(mesh_micro, mesh_micro.markers, 
                                               {0: alpha_micro_i, 1: alpha_micro_m})
    param_micro = {"lamb" : lamb_ , "mu": mu_ , "alpha" : alpha_}
    psi_mu = partial(psi_mu, param = param_micro)
        
    return MicroModel(mesh_micro, psi_mu, [1])

    
param={
'lx' : 2.0,
'ly' : 0.5,
'nx' : 40, 
'ny' : 10, 
'lamb': 1.0,
'mu' : 0.5,
'alpha' : 0.0,
'ty' : -0.005,
'clamped_bc' : 4, 
'load_bc' : 2,
'deg_u' : 1,
'deg_stress' : 0, 
'gdim': 2,
'dirichlet': [],
'neumann': [],
'msh_file' : "bar.msh",
'out_file' : "bar_multiscale.xdmf",
'msh_micro_file' : "./meshes/mesh_micro.msh",
'msh_micro_out_file' : "./meshes/mesh_micro.xdmf",
'create_mesh': True, 
'solver_atol' : 1e-12,
'solver_rtol' : 1e-12,
}

# for dirichlet and neumann: tuple of (physical group tag, direction, value)
param['dirichlet'].append((param['clamped_bc'], 0, 0.))
param['dirichlet'].append((param['clamped_bc'], 1, 0.))
param['neumann'].append((param['load_bc'], 1, param['ty']))


# macro-scale problem
gdim = param['gdim']
if(param['create_mesh']):
    ft.generate_rectangle_msh(param['msh_file'], 0.0, 0.0, param['lx'], param['ly'],
                              param['nx'], param['ny'])
    
msh =  ft.Mesh(param['msh_file'], MPI.COMM_WORLD, gdim = gdim)


lamb, mu, alpha = param['lamb'], param['mu'], param['alpha']
material_param = [fem.Constant(msh,lamb), fem.Constant(msh,mu), fem.Constant(msh, alpha)]


Uh = fem.functionspace(msh, ("CG", param['deg_u'], (param['gdim'],)))


# # create quadrature spaces: scalar, vectorial  (strain/stresses), and tensorial (tangents)
# def create_quadrature_spaces_mechanics(mesh, deg_q, qdim):
#     cell = mesh.ufl_cell()
#     q = "Quadrature"
#     QF = df.FiniteElement(q, cell, deg_q, quad_scheme="default")
#     QV = df.VectorElement(q, cell, deg_q, quad_scheme="default", dim=qdim)
#     QT = df.TensorElement(q, cell, deg_q, quad_scheme="default", shape=(qdim, qdim))
#     return [df.FunctionSpace(mesh, Q) for Q in [QF, QV, QT]]

n_strain = 3
n_tan = 6
W = CustomQuadratureSpace(msh, n_strain, degree_quad = 0)
W0 = CustomQuadratureSpace(msh, 1, degree_quad = 0)
Wtan = CustomQuadratureSpace(msh, n_tan, degree_quad = 0)

ng = W.scalar_space.dofmap.index_map.size_global
microModels = ng*[getMicroModel(param['msh_micro_file'])] # only valid because it's the same
hom = MicroMacro(W, Wtan, W.dxm, microModels)

dx = W.dxm
stress, tang = hom.stress, hom.tangent_op

def homogenisation_update(w, dw):
    hom.update(ft.symgrad_mandel(w))
    
callbacks = [homogenisation_update]



u = fem.Function(Uh)
v = ufl.TestFunction(Uh)
u_ = ufl.TrialFunction(Uh)


bcs_D = []
for bc in param['dirichlet']:
    bcs_D.append(ft.dirichletbc(fem.Constant(msh, bc[2]), bc[0], Uh.sub(bc[1])))
    
def F_ext(v):
    return sum([ ufl.inner(bc[2], v[bc[1]])*msh.ds(bc[0]) for bc in param['neumann']])

Celas = ft.Celas_mandel(param['lamb'], param['mu'])

nstraindim = 3

eps_var = ufl.variable(ft.symgrad_mandel(u))
psi = getPsi(eps_var, material_param)
stress = ufl.diff(psi, eps_var)
tangent = ufl.diff(stress, eps_var)

# res = ufl.inner(stress, ft.symgrad_mandel(v))*msh.dx - F_ext(v)
# jac = ufl.inner(ufl.dot(tangent, ft.symgrad_mandel(u_)), ft.symgrad_mandel(v))*msh.dx

q_.external_function = q_external

res = ufl.inner(q_, ft.symgrad_mandel(v))*msh.dx - F_ext(v)
J = ufl.derivative(res, T, u_)
# jac = ufl.inner(ufl.dot(tangent, ft.symgrad_mandel(u_)), ft.symgrad_mandel(v))*msh.dx

# res = ufl.inner(stress, ft.symgrad_mandel(v))*msh.dx - F_ext(v)
# jac = ufl.inner(ufl.dot(tangent, ft.symgrad_mandel(u_)), ft.symgrad_mandel(v))*msh.dx


neumann_term = sum([df.inner(df.Constant(value), v)*mesh.ds(flag) for value, flag in neumann])

res = df.inner(stress , ft.symgrad_mandel(v), )*dx - neumann_term
Jac = df.inner(tang(ft.symgrad_mandel(du)), ft.symgrad_mandel(v))*dx

# other manner 
u = df.Function(Uh)
problem = ft.CustomNonlinearProblem(-res, u, bcs, Jac)
solver = ft.CustomNonlinearSolver(problem, callbacks = callbacks)


problem = fem.petsc.NonlinearProblem(F_compiled, T, bcs_D, J = J_compiled)


solver = dolfinx.nls.petsc.NewtonSolver(msh.comm, problem)
# Set Newton solver options
solver.atol = param['solver_atol']
solver.rtol = param['solver_rtol']
# solver.convergence_criterion = "incremental"
solver.solve(u)
                                                       
# posprocessing
mesh = u.function_space.mesh


with io.XDMFFile(MPI.COMM_WORLD, param['out_file'], "w") as xdmf:
    xdmf.write_mesh(mesh)
    
with io.XDMFFile(MPI.COMM_WORLD, param['out_file'], "a") as xdmf:
    xdmf.write_function(u, 0)
    
    
print(np.linalg.norm(u.x.array))

