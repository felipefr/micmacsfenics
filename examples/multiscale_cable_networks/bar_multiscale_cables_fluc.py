#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:46:45 2025

@author: frocha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 12:03:37 2025

@author: frocha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: felipe rocha
Created on Thu Jun 13 16:56:55 2024

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
import subprocess
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from functools import partial 

whoami = subprocess.Popen("whoami", shell=True, stdout=subprocess.PIPE).stdout.read().decode()[:-1]
home = "/home/{0}/sources/".format(whoami) 
sys.path.append(home + "fetricksx")
sys.path.append(home + "micmacsfenicsx")
sys.path.append(home + "pyola")


import basix
import numpy as np
from dolfinx import fem, io
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from mpi4py import MPI

import fetricksx as ft
import micmacsfenicsx as mm

from micro_model_truss_fluctuation import MicroModelTrussFluctuation
import meshio

import toy_solver as pyola

def getMicroModel(param):
    mesh_vtk = meshio.read(param['mesh_file'])
    mesh_micro = pyola.Mesh(mesh_vtk.points[:,0:2], mesh_vtk.cells_dict['line'])
    mesh_micro.mark_boundary_nodes()
    param_truss = param
    param_truss['A'] = mesh_vtk.cell_data['A'][0] # otherwise is a list with an array
    
    return MicroModelTrussFluctuation(mesh_micro, param_truss)

    
param={
'lx' : 2.0,
'ly' : 0.5,
'nx' : 5, 
'ny' : 3, 
# 'ty' : -0.01,
'tx' : 0.02,
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
'micro_param_truss': {
    'mesh_file' : "cable_network.vtk",
    'model': 'truss', 
    'E' : 100.0,
    'eta' : 0.0,
    'solver': {'atol' : 1e-12, 'rtol' : 1e-12}
    },
'micro_param_cable': {
    'mesh_file' : "cable_network.vtk",
    'model': 'cable', 
    'E' : 100.0,
    'eta' : 0.0001,
    'solver': {'atol' : 1e-12, 'rtol' : 1e-12}
    }
}


# for dirichlet and neumann: tuple of (physical group tag, direction, value)
param['dirichlet'].append((param['clamped_bc'], 0, 0.))
param['dirichlet'].append((param['clamped_bc'], 1, 0.))
param['neumann'].append((param['load_bc'], 0, param['tx']))
# param['neumann'].append((param['load_bc'], 1, param['ty']))

n_strain = 4
n_tan = 10

# macro-scale problem
gdim = param['gdim']
if(param['create_mesh']):
    ft.generate_rectangle_mesh(param['msh_file'], 0.0, 0.0, param['lx'], param['ly'],
                              param['nx'], param['ny'])
    
msh =  ft.Mesh(param['msh_file'], MPI.COMM_WORLD, gdim = gdim)

Uh = fem.functionspace(msh, ("CG", param['deg_u'], (param['gdim'],)))
u = fem.Function(Uh)
v = ufl.TestFunction(Uh)
u_ = ufl.TrialFunction(Uh)

W = ft.CustomQuadratureSpace(msh, n_strain, degree_quad = 0)
Wtan = ft.CustomQuadratureSpace(msh, n_tan, degree_quad = 0)

ng = W.scalar_space.dofmap.index_map.size_global 
# microModels = ng*[getMicroModel(param['micro_param'])] # with RVE repetition
micromodels_truss = [getMicroModel(param['micro_param_truss']) for i in range(ng)]  # without RVE repetition
micromodels_cable = [getMicroModel(param['micro_param_cable']) for i in range(ng)]

hom = mm.MicroMacro(W, Wtan, W.dxm, micromodels_truss)

bcs_D = []
for bc in param['dirichlet']:
    bcs_D.append(ft.dirichletbc(fem.Constant(msh, bc[2]), bc[0], Uh.sub(bc[1])))
    
def F_ext(v):
    return sum([ ufl.inner(bc[2], v[bc[1]])*msh.ds(bc[0]) for bc in param['neumann']])

dx = W.dxm
stress, tangent = hom.stress, hom.tangent_op
hom.set_track_strain(ft.grad_unsym(u))

res = ufl.inner(stress, ft.grad_unsym(v))*dx - F_ext(v)
jac = ufl.inner(tangent(ft.grad_unsym(u_)), ft.grad_unsym(v))*dx

# other manner 
start = timer()
problem = ft.CustomNonlinearProblem(res, u, bcs_D, jac)
solver = ft.CustomNonlinearSolver(problem, callbacks = [hom.update])
solver.solve(report = True, Nitermax = 50, omega = 1.0)
end = timer()
print("time: ", end - start)
print(np.linalg.norm(u.x.array))

for i in range(ng):
    micromodels_cable[i].u = micromodels_truss[i].u 
    
hom.micromodels = micromodels_cable

start = timer()
problem = ft.CustomNonlinearProblem(res, u, bcs_D, jac)
solver = ft.CustomNonlinearSolver(problem, callbacks = [hom.update])
solver.solve(report = True, Nitermax = 50, omega = 0.8)
end = timer()

# posprocessing
mesh = u.function_space.mesh


with io.XDMFFile(MPI.COMM_WORLD, param['out_file'], "w") as xdmf:
    xdmf.write_mesh(mesh)
    
with io.XDMFFile(MPI.COMM_WORLD, param['out_file'], "a") as xdmf:
    xdmf.write_function(u, 0)
    
    
print(np.linalg.norm(u.x.array))

