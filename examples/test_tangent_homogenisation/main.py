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
    contrast = 10.0
    bndModel_micro = 'lin'
    vf = 0.124858 # volume fraction of inclusions vf*c_i + (1-vf)*c_m
    
    psi_mu = ft.psi_ciarlet_F
    
    lamb_micro_m = lamb_ref/( 1-vf + contrast*vf)
    mu_micro_m = mu_ref/( 1-vf + contrast*vf)
    lamb_micro_i = contrast*lamb_micro_m
    mu_micro_i = contrast*mu_micro_m
    
    mesh_micro = ft.Mesh(mesh_micro_name)
    
    lamb_ = ft.getMultimaterialExpression(np.array([lamb_micro_i, lamb_micro_m]).reshape((2,1)), mesh_micro, op = 'cpp')
    mu_ = ft.getMultimaterialExpression(np.array([mu_micro_i, mu_micro_m]).reshape((2,1)), mesh_micro, op = 'cpp')
    
    param_micro = {"lamb" : lamb_[0] , "mu": mu_[0]} 
    
    psi_mu = partial(psi_mu, param = param_micro)
        
    if(gdim==3):
        return mm.MicroConstitutiveModelFiniteStrain3d(mesh_micro, psi_mu, bndModel_micro)
    elif(gdim==2):
        return mm.MicroConstitutiveModelFiniteStrain(mesh_micro, psi_mu, bndModel_micro)


def get_tangent_pertubation_forward(Gmacro, micromodel, tau = 1e-6):
    micromodel.setUpdateFlag(False)
    micromodel.restart_initial_guess()
    stress_ref = micromodel.getStress(Gmacro)
    n = len(Gmacro)
    base_canonic = np.eye(n)
    Atang = np.zeros((n,n))
    
    for j in range(n):
        micromodel.restart_initial_guess()
        micromodel.setUpdateFlag(False)
        stress_per = micromodel.getStress(Gmacro + tau*base_canonic[j,:])
        Atang[:,j] = (stress_per - stress_ref)/tau 
    
    return Atang

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

def get_error_tang(Gmacro, micromodel, tau = 1e-6, method = 'DF', A_ref = None):    
    if(type(A_ref) == type(None)):
        micromodel.setUpdateFlag(False)
        micromodel.restart_initial_guess()
        A_ref = micromodel.getTangent(Gmacro)
        
    get_tangent = {'DF': get_tangent_pertubation_forward, 
                   'CD': get_tangent_pertubation_central_difference}[method]
    
    A_pert = get_tangent(Gmacro, micromodel, tau)

    return np.linalg.norm(A_pert-A_ref)/np.linalg.norm(A_ref)


def study_tau(micromodel):

    Gmacro = np.array([0.1,0.3,-0.2,0.3])
    micromodel.restart_initial_guess()
    micromodel.setUpdateFlag(False)
    A_ref = micromodel.getTangent(Gmacro)
    
    tau_list = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
    error = []
    error_CD = []
    
    for tau in tau_list:
        error.append(get_error_tang(Gmacro, micromodel, tau, 'DF', A_ref))
        error_CD.append(get_error_tang(Gmacro, micromodel, tau, 'CD', A_ref))
    
    
    plt.plot(tau_list, error,'-o', label = 'pertubation')
    plt.plot(tau_list, error_CD,'-o',label = 'pertubation CD')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('tau')
    plt.grid()
    plt.ylabel('relative error')

if __name__ == "__main__":
    
    
    gdim = 2
    clampedFlag = 1 # left
    
    
    mesh_micro_name_geo = './meshes/{0}d/mesh_micro.geo'.format(gdim)
    mesh_micro_name = './meshes/{0}d/mesh_micro.xdmf'.format(gdim)
    
    micromodel = getMicroModel(mesh_micro_name, gdim)  
    
    
    
    # Gmacro = np.array([0.1,0.3,-0.2,0.3])
    # micromodel.restart_initial_guess()
    # micromodel.setUpdateFlag(False)
    # A_ref = micromodel.getTangent(Gmacro)
    # print(A_ref)
    
    # Gmacro = np.array([0.1,0.3,-0.2,0.3])
    # micromodel.restart_initial_guess()
    # micromodel.solve_microproblem(Gmacro)
    # A_ref = micromodel.homogenise_tangent()
    # print(A_ref)
    
    # A_per = get_tangent_pertubation_central_difference(Gmacro, micromodel)
    # print(A_per)
    
    # study_tau(micromodel)
    
    nstrain = gdim**2
    
    tau = 1e-7

    time_ref = 0.0
    time_DF = 0.0 
    time_CD = 0.0
    
    Nsteps = 80
    dload = 0.005
    Gmacro = np.zeros((Nsteps,nstrain))
    Gmacro[0,:] = dload*np.random.randn(len(Gmacro[0,:]))
    
    for i in range(Nsteps-1):
        Gmacro[i+1,:] = Gmacro[i,:] + dload*np.random.randn(len(Gmacro[0,:]))

    start = timer()  
    micromodel.restart_initial_guess()
    for i in range(Nsteps):
        micromodel.setUpdateFlag(False)
        A_ref = micromodel.getTangent(Gmacro[i,:])
    end = timer()
    time_ref += (end-start)
        
    start = timer()  
    micromodel.restart_initial_guess()
    for i in range(Nsteps):
        micromodel.setUpdateFlag(False)
        A_DF = get_tangent_pertubation_forward(Gmacro[i,:], micromodel, tau)
    end = timer()
    time_DF += (end-start)
        
    start = timer()  
    micromodel.restart_initial_guess()
    for i in range(Nsteps):
        micromodel.setUpdateFlag(False)
        A_CD = get_tangent_pertubation_central_difference(Gmacro[i,:], micromodel, tau)
    end = timer()
    time_CD += (end-start)

    print("time ref", time_ref)
    print("speedup CD:" , time_CD/time_ref)
    print("speedup DF:" , time_DF/time_ref)