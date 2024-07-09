#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:00:01 2024
@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2021-2024, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or <felipe.f.rocha@gmail.com>
"""

"""
Bar problem given a constitutive law (single-scale):
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and traction on the
right. We use an isotropic linear material, given two lamé parameters.
"""


import sys
from timeit import default_timer as timer
from functools import partial 

sys.path.append("/home/felipe/sources/fetricksx")
sys.path.append("/home/felipe/sources/micmacsfenicsx/core")

import numpy as np
from dolfinx import fem, io
import ufl
from mpi4py import MPI

import fetricksx as ft
import micmacsfenicsx as mm

def getMicroModel(param):
        
    mesh_micro_name = param['msh_file']
    lamb_ref = param['lamb_ref']
    mu_ref = param['mu_ref']
    contrast = param['contrast']
#    bnd = param['bnd']
    vf = param['vf'] # volume fraction of inclusions vf*c_i + (1-vf)*c_m
    psi_mu = param['psi_mu']
    alpha_m = param['alpha_m']
    alpha_i = param['alpha_m']

    lamb_m = lamb_ref/( 1-vf + contrast*vf)
    mu_m = mu_ref/( 1-vf + contrast*vf)
    lamb_i = contrast*lamb_m
    mu_i = contrast*mu_m
    
    mesh_micro = ft.Mesh(mesh_micro_name)
    
    alpha_ = ft.create_piecewise_constant_field(mesh_micro, mesh_micro.markers, 
                                               {0: alpha_i, 1: alpha_m})
    lamb_ = ft.create_piecewise_constant_field(mesh_micro, mesh_micro.markers, 
                                               {0: lamb_i, 1: lamb_m})
    mu_ = ft.create_piecewise_constant_field(mesh_micro, mesh_micro.markers, 
                                               {0: mu_i, 1: mu_m})
    param_micro = {"lamb" : lamb_ , "mu": mu_ , "alpha" : alpha_}
    psi_mu = partial(psi_mu, param = param_micro)
        
    return mm.MicroModel(mesh_micro, psi_mu, bnd_flags=[0], solver_param = param['solver'])

    
param={
'lx' : 2.0,
'ly' : 0.5,
'nx' : 10, 
'ny' : 3, 
'ty' : -0.01,
'clamped_bc' : 4, 
'load_bc' : 2,
'deg_u' : 1,
'deg_stress' : 0, 
'gdim': 2,
'dirichlet': [],
'neumann': [],
'msh_file' : "bar.msh",
'out_file' : "bar_multiscale.xdmf",
'create_mesh': True, 
'solver_atol' : 1e-12,
'solver_rtol' : 1e-12,
'micro_param': {
    'msh_file' : "../meshes/mesh_micro.geo",
    'msh_out_file' : "../meshes/mesh_micro.xdmf",
    'lamb_ref' : 10.0,
    'mu_ref' : 5.0,
    'contrast' : 1.0, # change contrast to have true heterogeneous microstructue
    'bnd' : 'lin',
    'vf' : 0.124858,
    'psi_mu' : ft.psi_hookean_nonlinear_lame,
    'alpha_m' : 10000.0,
    'alpha_i' : 10000.0,
    'solver': {'atol' : 1e-12, 'rtol' : 1e-12}
    }
}

# for dirichlet and neumann: tuple of (physical group tag, direction, value)
param['dirichlet'].append((param['clamped_bc'], 0, 0.))
param['dirichlet'].append((param['clamped_bc'], 1, 0.))
param['neumann'].append((param['load_bc'], 1, param['ty']))

n_strain = 3
n_tan = 6

# macro-scale problem
gdim = param['gdim']
if(param['create_mesh']):
    ft.generate_rectangle_msh(param['msh_file'], 0.0, 0.0, param['lx'], param['ly'],
                              param['nx'], param['ny'])
    
msh =  ft.Mesh(param['msh_file'], MPI.COMM_WORLD, gdim = gdim)

Uh = fem.functionspace(msh, ("CG", param['deg_u'], (param['gdim'],)))
u = fem.Function(Uh)
v = ufl.TestFunction(Uh)
u_ = ufl.TrialFunction(Uh)

W = ft.CustomQuadratureSpace(msh, n_strain, degree_quad = 0)
Wtan = ft.CustomQuadratureSpace(msh, n_tan, degree_quad = 0)

ng = W.scalar_space.dofmap.index_map.size_global # only valid because it's one RVE per element
microModels = ng*[getMicroModel(param['micro_param'])] 
hom = mm.MicroMacroExternalOperator(ft.symgrad_mandel(u), W, Wtan, microModels)

# another possible implementation (uncomment the class on the same file of the original one)
# Wstress_tan = ft.CustomQuadratureSpace(msh, n_tan + n_strain, degree_quad = 0)
# hom = mm.MicroMacroExternalOperator(ft.symgrad_mandel(u), Wstress_tan, microModels)

bcs_D = []
for bc in param['dirichlet']:
    bcs_D.append(ft.dirichletbc(fem.Constant(msh, bc[2]), bc[0], Uh.sub(bc[1])))
    
def F_ext(v):
    return sum([ ufl.inner(bc[2], v[bc[1]])*msh.ds(bc[0]) for bc in param['neumann']])

dx = W.dxm

res_ext = ufl.inner(hom.stress, ft.symgrad_mandel(v))*dx - F_ext(v)
J_ext = ufl.inner(hom.tangent_op(ft.symgrad_mandel(u_)), ft.symgrad_mandel(v))*dx

res_ext, J_ext = hom.register_forms(res_ext, J_ext)

start = timer()
problem = ft.CustomNonlinearProblem(res_ext, u, bcs_D, J_ext)
solver = ft.CustomNonlinearSolver(problem, callbacks=[hom.update])
solver.solve(report = True)
end = timer()
print("time: ", end - start)

# # posprocessing
mesh = u.function_space.mesh

with io.XDMFFile(MPI.COMM_WORLD, param['out_file'], "w") as xdmf:
    xdmf.write_mesh(mesh)
    
with io.XDMFFile(MPI.COMM_WORLD, param['out_file'], "a") as xdmf:
    xdmf.write_function(u, 0)
    
    
print(np.linalg.norm(u.x.array))

