#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:09:03 2025

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
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from functools import partial 

import basix
import numpy as np
from dolfinx import fem, io
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from mpi4py import MPI

sys.path.append("/home/frocha/sources/fetricksx")
import fetricksx as ft
sys.path.append("/home/frocha/sources/micmacsfenicsx")
import micmacsfenicsx as mm

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType  # type: ignore
import dolfinx_mpc
import dolfinx_mpc.utils
from dolfinx.io.gmshio import model_to_mesh
from dolfinx_mpc import LinearProblem as LinearProblemMPC

# petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

class CustomNonlinearProblemMPC:
    def __init__(self, res, u, mpc, bcs, jac):
        self.res = res
        self.u = u
        self.bcs = bcs
        self.mpc = mpc
        self.jac = jac


# # Based on https://bleyerj.github.io/comet-fenicsx/tours/nonlinear_problems/plasticity/plasticity.html
# class CustomTangentProblemMPC(LinearProblemMPC): 
#     def assemble_rhs(self, u=None):
#         """Assemble right-hand side and lift Dirichlet bcs.

#         Parameters
#         ----------
#         u : dolfinx.fem.Function, optional
#             For non-zero Dirichlet bcs u_D, use this function to assemble rhs with the value u_D - u_{bc}
#             where u_{bc} is the value of the given u at the corresponding. Typically used for custom Newton methods
#             with non-zero Dirichlet bcs.
#         """

#         # Assemble rhs
#         with self._b.localForm() as b_loc:
#             b_loc.set(0)
#         fem.petsc.assemble_vector(self._b, self._L)

#         # Apply boundary conditions to the rhs
#         x0 = [] if u is None else [u.x]
#         fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.bcs], x0=x0)
#         self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#         x0 = None if u is None else u.x
#         fem.petsc.set_bc(self._b, self.bcs, x0)

#     def assemble_lhs(self):
#         self._A.zeroEntries()
#         fem.petsc.assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
#         self._A.assemble()

#     def solve_system(self):
#         # Solve linear system and update ghost values in the solution
#         self._solver.solve(self._b, self._x)
#         self.u.x.scatter_forward()

class CustomNonlinearSolverMPC:

    # bcs: original object
    def __init__(self, problem, callbacks = [], u0_satisfybc = False): 
        
        self.problem = problem
        self.callbacks = callbacks
        self.u0_satisfybc = u0_satisfybc
    
        self.du = fem.Function(self.problem.u.function_space)
        
        # self.tangent_problem = CustomTangentProblemMPC(
        self.tangent_problem = LinearProblemMPC(
        self.problem.jac, -self.problem.res,
        u=self.du,
        bcs=self.problem.bcs,
        mpc= self.problem.mpc, 
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"})
        
    def reset_bcs(self, bcs):
        self.problem.reset_bcs(bcs)
        
    def solve(self, Nitermax = 10, tol = 1e-8, report = False):
        # compute the residual norm at the beginning of the load step
        self.call_callbacks()
        nRes = []
        
        #self.tangent_problem.assemble_rhs()
        nRes.append(self.tangent_problem._b.norm())
        if(nRes[0]<tol): 
            nRes[0] = 1.0
        self.du.x.array[:] = 0.0

        niter = 0
        while nRes[niter] / nRes[0] > tol and niter < Nitermax:
            # solve for the displacement correction
            # self.tangent_problem.assemble_lhs()
            # self.tangent_problem.solve_system()
            self.tangent_problem.solve()

            # update the displacement increment with the current correction
            self.problem.u.x.petsc_vec.axpy(1, self.du.x.petsc_vec)  # Du = Du + 1*du
            self.problem.u.x.scatter_forward()
            self.call_callbacks()
            
            # self.tangent_problem.assemble_rhs()
            nRes.append(self.tangent_problem._b.norm())
            
            niter += 1
            if(report):
                print(" Residual:", nRes[-1]/nRes[0])

        return nRes

    def call_callbacks(self):
        [foo(self.problem.u, self.du) for foo in self.callbacks]




def get_micro_psi(mesh_micro, param):
        
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
    

    
    alpha_ = ft.create_piecewise_constant_field(mesh_micro, mesh_micro.markers, 
                                               {0: alpha_i, 1: alpha_m})
    lamb_ = ft.create_piecewise_constant_field(mesh_micro, mesh_micro.markers, 
                                               {0: lamb_i, 1: lamb_m})
    mu_ = ft.create_piecewise_constant_field(mesh_micro, mesh_micro.markers, 
                                               {0: mu_i, 1: mu_m})
    
    param_micro = {"lamb" : lamb_ , "mu": mu_ , "alpha" : alpha_}
    psi_mu = partial(psi_mu, param = param_micro)
        
    return psi_mu 
# mm.MicroModelMinimallyConstrained(mesh_micro, psi_mu, bnd_flags=[0], solver_param = param['solver'])

    
param={
'micro_param': {
    'msh_file' : "mesh_micro.geo",
    'msh_out_file' : "mesh_micro.xdmf",
    'lamb_ref' : 10.0,
    'mu_ref' : 5.0,
    'contrast' : 10.0, # change contrast to have true heterogeneous microstructue
    'bnd' : 'lin',
    'vf' : 0.124858,
    'psi_mu' : ft.psi_hookean_nonlinear_lame,
    'alpha_m' : 0.0,
    'alpha_i' : 0.0,
    'solver': {'atol' : 1e-12, 'rtol' : 1e-12}
    }
}

mesh_micro = ft.Mesh(param['micro_param']['msh_file'])

psi_model = get_micro_psi(mesh_micro, param['micro_param']) 



            # self.W0e = basix.ufl.element("DG", self.basix_cell, degree= self.degree_quad)
            # self.We = basix.ufl.element("DG", self.basix_cell, degree= self.degree_quad, shape = (dim,))
            # self.space = fem.functionspace(self.mesh, self.We)       
            # self.scalar_space = fem.functionspace(self.mesh, self.W0e)



mesh = mesh_micro
tdim = mesh.topology.dim
gdim = mesh.geometry.dim
nmandel = int(tdim*(tdim + 1)/2)   
solver_param = param['micro_param']['solver']

psi_mu = psi_model

Uh = fem.functionspace(mesh, ("CG", 1, (tdim,)))

X = mesh.geometry.x
corners = np.array([[X[:,0].min(), X[:,1].min()], [X[:,0].max(), X[:,1].min()], 
                    [X[:,0].max(), X[:,1].max()], [X[:,0].min(), X[:,1].max()]])
                    
a1 = corners[1, :] - corners[0, :]  # first vector generating periodicity
a2 = corners[3, :] - corners[0, :]  # second vector generating periodicity


def periodic_relation_left_right(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - a1[0]
    out_x[1] = x[1] - a1[1]
    out_x[2] = x[2]
    return out_x


def periodic_relation_bottom_top(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - a2[0]
    out_x[1] = x[1] - a2[1]
    out_x[2] = x[2]
    return out_x

point_dof = fem.locate_dofs_geometrical(
    Uh, lambda x: np.isclose(x[0], corners[0,0]) & np.isclose(x[1], corners[0,1])
)
bcs = [fem.dirichletbc(np.zeros((gdim,)), point_dof, Uh)]



mpc = dolfinx_mpc.MultiPointConstraint(Uh)
mpc.create_periodic_constraint_topological(
    Uh, mesh.facets, 2, periodic_relation_left_right, bcs
)
mpc.create_periodic_constraint_topological(
    Uh, mesh.facets, 3, periodic_relation_bottom_top, bcs
)
mpc.finalize()

V = mpc.function_space # redefinition

dy = mesh.dx
vol = ft.integral(fem.Constant(mesh,1.0), dy, mesh, ())
inv_vol = vol**-1

y = ufl.SpatialCoordinate(mesh)
Eps = fem.Constant(mesh, np.array([0., 0., 0.], dtype = ScalarType))  # just placeholder
Eps_kl = fem.Constant(mesh, np.array([0., 0., 0.], dtype = ScalarType))  # just placeholder

duh = ufl.TrialFunction(V)            # Incremental displacement
vh  = ufl.TestFunction(V)             # Test function
uh  = fem.Function(V)                 # Displacement from previous iteration
ukl = fem.Function(V)
uh.x.array[:] = 0.0

eps = Eps  + ft.symgrad_mandel(uh)            # Deformation gradient
eps_var = ufl.variable(eps)

sigmu, Cmu = ft.get_stress_tang_from_psi(psi_mu, eps_var, eps_var) 

Jac = ufl.inner(ufl.dot( Cmu, ft.symgrad_mandel(duh)), ft.symgrad_mandel(vh))*dy

# generic residual for either fluctuation and canonical problem
flag_linres = fem.Constant(mesh, 1.0)
flag_nonlinres = fem.Constant(mesh, 0.0)
r = flag_nonlinres*sigmu + flag_linres*ufl.dot( Cmu, Eps_kl)

Res = ufl.inner(r , ft.symgrad_mandel(vh))*dy
        
microproblem = CustomNonlinearProblemMPC(Res, uh, mpc, bcs, Jac)
microsolver = CustomNonlinearSolverMPC(microproblem)

uh.x.array[:] = 0.0
Eps.value = np.array([0.01,0.0,0.001])
microsolver.solve()
stress_hom = ft.integral(sigmu, dy, mesh, (nmandel,))
print(stress_hom)


micro_model = mm.MicroModel(mesh_micro, psi_mu, bnd_flags=[0], solver_param = solver_param)
micro_model.flag_linres.value = 1.0
micro_model.flag_nonlinres.value = 0.0
micro_model.restart_initial_guess()
micro_model.Eps.value = Eps.value
micro_model.microsolver.solve()
stress_hom2 = micro_model.homogenise_stress()
print(stress_hom2)