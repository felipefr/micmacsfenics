#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 23:23:42 2024

@author: felipe
"""

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

import basix
import numpy as np
from dolfinx import fem, io
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from mpi4py import MPI

import fetricksx as ft

from external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

# elastic parameters
def getPsi(e, param):
    tr_e = ft.tr_mandel(e)
    e2 = ufl.inner(e, e)

    lamb, mu, alpha = param
    
    return (0.5*lamb*(1.0 + 0.5*alpha*(tr_e**2))*(tr_e**2) + mu*(1 + 0.5*alpha*e2)*(e2))

    
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

u = fem.Function(Uh)
v = ufl.TestFunction(Uh)
u_ = ufl.TrialFunction(Uh)


bcs_D = []
for bc in param['dirichlet']:
    bcs_D.append(ft.dirichletbc(fem.Constant(msh, bc[2]), bc[0], Uh.sub(bc[1])))
    
def F_ext(v):
    return sum([ ufl.inner(bc[2], v[bc[1]])*msh.ds(bc[0]) for bc in param['neumann']])

# Wh = fem.functionspace(msh, ("CG", param['deg_u'], (param['gdim'],)))
T = fem.Function(Uh)
e = ft.symgrad_mandel(T)
T.interpolate(lambda x: np.array([x[0] ** 2 + x[1], x[1]]))
quadrature_degree = 2
Qe = basix.ufl.quadrature_element(msh.topology.cell_name(), degree=quadrature_degree, value_shape=(3,))
Q = fem.functionspace(msh, Qe)
# dx = ufl.Measure("dx", metadata={"quadrature_scheme": "default", "quadrature_degree": quadrature_degree})
q_ = FEMExternalOperator(T, e, function_space=Q)


Celas = ft.Celas_mandel(param['lamb'], param['mu'])

nstraindim = 3

def q_impl(theta, g):
    G = g.reshape((-1,  nstraindim))
    output = G @ Celas
    return output.reshape(-1)


# def dqdT_impl(T, sigma):
#     num_cells = T.shape[0]
#     sigma_ = sigma.reshape((num_cells, -1, gdim))
#     output = B * (k(T) ** 2)[:, :, np.newaxis] * sigma_
#     return output.reshape(-1)


def dqdsigma_impl(T, sigma):
    output = np.zeros((T.shape[0]*T.shape[1], 9))
    for i in range(len(output)):
        output[i,:] = Celas.flatten()
    return output.reshape(-1)

eps_var = ufl.variable(ft.symgrad_mandel(u))
psi = getPsi(eps_var, material_param)
stress = ufl.diff(psi, eps_var)
tangent = ufl.diff(stress, eps_var)


def q_external(derivatives):
    if derivatives == (0, 0):
        return q_impl
    elif derivatives == (1, 0):
        return dqdsigma_impl
    elif derivatives == (0, 1):
        return dqdsigma_impl
    else:
        return NotImplementedError

# res = ufl.inner(stress, ft.symgrad_mandel(v))*msh.dx - F_ext(v)
# jac = ufl.inner(ufl.dot(tangent, ft.symgrad_mandel(u_)), ft.symgrad_mandel(v))*msh.dx

q_.external_function = q_external

res = ufl.inner(q_, ft.symgrad_mandel(v))*msh.dx - F_ext(v)
J = ufl.derivative(res, T, u_)
# jac = ufl.inner(ufl.dot(tangent, ft.symgrad_mandel(u_)), ft.symgrad_mandel(v))*msh.dx

# res = ufl.inner(stress, ft.symgrad_mandel(v))*msh.dx - F_ext(v)
# jac = ufl.inner(ufl.dot(tangent, ft.symgrad_mandel(u_)), ft.symgrad_mandel(v))*msh.dx


F_replaced, F_external_operators = replace_external_operators(res)
J_expanded = ufl.algorithms.expand_derivatives(J)
J_replaced, J_external_operators = replace_external_operators(J_expanded)
evaluated_operands = evaluate_operands(F_external_operators)
_ = evaluate_external_operators(F_external_operators, evaluated_operands)

_ = evaluate_external_operators(J_external_operators, evaluated_operands)
F_compiled = fem.form(F_replaced)
J_compiled = fem.form(J_replaced)
b_vector = fem.assemble_vector(F_compiled)
A_matrix = fem.assemble_matrix(J_compiled)

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

