#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:25:47 2023

@author: ffiguere
"""

# Known bugs:
# 1) When trying to generate mesh: create_mesh == True

#     self.exportMeshHDF5(savefile, optimize_storage)

#   File ~/miniforge3/envs/fenics_2019/lib/python3.8/site-packages/fetricks/fenics/mesh/wrapper_gmsh.py:68 in exportMeshHDF5
#     self.__determine_geometry_types(mesh_msh)

#   File ~/miniforge3/envs/fenics_2019/lib/python3.8/site-packages/fetricks/fenics/mesh/wrapper_gmsh.py:51 in __determine_geometry_types
#     self.facet_type, self.cell_type = mesh_msh.cells_dict.keys()

# AttributeError: 'Mesh' object has no attribute 'cells_dict'


import sys, os
import subprocess
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'

whoami = subprocess.Popen("whoami", shell=True, stdout=subprocess.PIPE).stdout.read().decode()[:-1]
home = "/home/{0}/sources/".format(whoami) 
sys.path.append(home + "ddfenics")
sys.path.append(home + "fetricks")
sys.path.append(home + "micmacsfenics")

import dolfin as df
import ufl
import numpy as np
import ddfenics as dd
import fetricks as ft 
import micmacsfenics as mm
from timeit import default_timer as timer
from functools import partial
import matplotlib.pyplot as plt

df.parameters["form_compiler"]["representation"] = 'uflacs'
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3"

import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

comm = df.MPI.comm_world

def getMicroModel(mesh_micro_name= "../meshes/mesh_micro.xdmf", gdim=2):
    # mesh_micro_name = 'meshes/mesh_micro.xdmf'
    lamb_ref = 432.099
    mu_ref = 185.185
    bndModel_micro = ['MRHO', [1,2]]
    
    psi_mu = ft.psi_hookean_nonlinear_lame
    mesh_micro = ft.Mesh(mesh_micro_name)
    
    param_micro = {"lamb" : df.Constant(lamb_ref) , "mu": df.Constant(mu_ref), "alpha": df.Constant(0.0)} 
    
    psi_mu = partial(psi_mu, param = param_micro)
        
    return mm.MicroConstitutiveModelHighOrder(mesh_micro, psi_mu, bndModel_micro)

def get_tangent_pertubation_central_difference(Gmacro, micromodel, tau = 1e-6):
    n = len(Gmacro)
    base_canonic = np.eye(n)
    Atang = np.zeros((n,n))
    
    for j in range(n):
        micromodel.restart_initial_guess()
        micromodel.setUpdateFlag(False)
        stress_per_p = micromodel.getStress(Gmacro + 0.5*tau*base_canonic[j,:])
        micromodel.restart_initial_guess()
        micromodel.setUpdateFlag(False)
        stress_per_m = micromodel.getStress(Gmacro - 0.5*tau*base_canonic[j,:])
        
        Atang[:,j] = (stress_per_p - stress_per_m)/tau 
    
    return Atang

if __name__ == "__main__":
    
    
    gdim = 2
    clampedFlag = 1 # left
    
    
    # mesh_micro_name_geo = './meshes/mesh_hexagon.geo'.format(gdim)
    # mesh_micro_name = './meshes/mesh_hexagon.xdmf'.format(gdim)
    
    mesh_micro_name_geo = './meshes/mesh_square.geo'.format(gdim)
    mesh_micro_name = './meshes/mesh_square.xdmf'.format(gdim)
    
    gmsh_mesh = ft.GmshIO(mesh_micro_name_geo, gdim)
    gmsh_mesh.write("xdmf")
    
    
    micromodel = getMicroModel(mesh_micro_name, gdim)  
    
    nstrain = 3
    Gmacro = np.array([0.1,0.3,-0.2])
    micromodel.setCanonicalproblem()
    micromodel.restart_initial_guess()
    micromodel.setUpdateFlag(False)
    #A_ref = micromodel.getTangent(Gmacro)
    A_per = get_tangent_pertubation_central_difference(Gmacro, micromodel)
    A_mp = micromodel.compute_tangent_multiphenics()
    A_lt = micromodel.compute_tangent_localisation_tensors_full_notation()
    #print(A_ref)
    print(A_per)
    print(A_mp)
    print(A_lt)
    print(np.linalg.norm(A_mp - A_lt)/np.linalg.norm(A_lt))
    
    B_lt = micromodel.compute_hypertangent()