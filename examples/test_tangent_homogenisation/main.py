#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:25:47 2023

@author: ffiguere
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


import basix
import numpy as np
from dolfinx import fem, io
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from mpi4py import MPI

import fetricksx as ft
import micmacsfenicsx as mm

def getMicroModel(param):
        
    mesh_micro_name = param['msh_file']
    lamb_ref = param['lamb_ref']
    mu_ref = param['mu_ref']
    contrast = param['contrast']
    # bndModel_micro = param['bndModel'] # lin for the moment
    vf = param['vf'] # volume fraction of inclusions vf*c_i + (1-vf)*c_m
    psi_mu = param['psi_mu']

    lamb_m = lamb_ref/( 1-vf + contrast*vf)
    mu_m = mu_ref/( 1-vf + contrast*vf)
    lamb_i = contrast*lamb_m
    mu_i = contrast*mu_m
    
    mesh_micro = ft.Mesh(mesh_micro_name)
    
    lamb_ = ft.create_piecewise_constant_field(mesh_micro, mesh_micro.markers, 
                                               {0: lamb_i, 1: lamb_m})
    mu_ = ft.create_piecewise_constant_field(mesh_micro, mesh_micro.markers, 
                                               {0: mu_i, 1: mu_m})
    
    param_micro = {"lamb" : lamb_ , "mu": mu_}
    psi_mu = partial(psi_mu, param = param_micro)
        
    return mm.MicroModelFiniteStrain(mesh_micro, psi_mu, bnd_flags=[0], solver_param = param['solver'])


def get_tangent_pertubation_forward(Gmacro, micromodel, tau = 1e-6):
#    micromodel.setUpdateFlag(False)
    micromodel.restart_initial_guess()
    micromodel.solve_microproblem(Gmacro)
    stress_ref = micromodel.homogenise_stress()
    n = len(Gmacro)
    base_canonic = np.eye(n)
    Atang = np.zeros((n,n))
    
    for j in range(n):
        micromodel.restart_initial_guess()
        micromodel.solve_microproblem(Gmacro + tau*base_canonic[j,:])
        stress_per = micromodel.homogenise_stress()
        Atang[:,j] = (stress_per - stress_ref)/tau 
    
    return Atang

def get_tangent_pertubation_central_difference(Gmacro, micromodel, tau = 1e-6):
    n = len(Gmacro)
    base_canonic = np.eye(n)
    Atang = np.zeros((n,n))
    
    for j in range(n):
        micromodel.restart_initial_guess()
        micromodel.solve_microproblem(Gmacro + 0.5*tau*base_canonic[j,:])
        stress_per_p = micromodel.homogenise_stress()
        micromodel.restart_initial_guess()
        micromodel.solve_microproblem(Gmacro - 0.5*tau*base_canonic[j,:])
        stress_per_m = micromodel.homogenise_stress()
        
        Atang[:,j] = (stress_per_p - stress_per_m)/tau 
    
    return Atang

def get_error_tang(Gmacro, micromodel, tau = 1e-6, method = 'DF', A_ref = None):
    
    if(type(A_ref) == type(None)):
        micromodel.restart_initial_guess()
        micromodel.solve_microproblem(Gmacro)
        A_ref = micromodel.homogenise_tangent()
        
    get_tangent = {'DF': get_tangent_pertubation_forward, 
                   'CD': get_tangent_pertubation_central_difference}[method]
    
    A_pert = get_tangent(Gmacro, micromodel, tau)

    return np.linalg.norm(A_pert-A_ref)/np.linalg.norm(A_ref)


def study_tau(micromodel):
    
    Gmacro = np.array([0.1,0.3,-0.2,0.3])
    micromodel.restart_initial_guess()
    micromodel.solve_microproblem(Gmacro)
    A_ref = micromodel.homogenise_tangent()
    
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
    
    param={
        'gdim': 2,
        'msh_file' : "./meshes/mesh_micro.geo",
        'msh_out_file' : "./meshes/mesh_micro.msh",
        'lamb_ref' : 432.099,
        'mu_ref' : 185.185,
        'contrast' : 10.0, # change contrast to have true heterogeneous microstructue
        'bnd' : 'lin',
        'vf' : 0.124858,
        'psi_mu' : ft.psi_ciarlet_F,
        'solver': {'atol' : 1e-8, 'rtol' : 1e-8, 'poly_order': 1}
    }
    
    
    micromodel = getMicroModel(param)
    
    # Gmacro = np.array([0.1,0.3,-0.2,0.3])
    # micromodel.restart_initial_guess()
    # micromodel.solve_microproblem(Gmacro)
    # A_ref = micromodel.homogenise_tangent()
    # print(A_ref)
    
    # A_per = get_tangent_pertubation_central_difference(Gmacro, micromodel)
    # print(A_per)
    
    
    # study_tau(micromodel)
    
    tau = 1e-7

    time_ref = 0.0
    time_DF = 0.0 
    time_CD = 0.0
    
    Nsteps = 80
    dload = 0.005
    Gmacro = np.zeros((Nsteps,micromodel.nstrain))
    Gmacro[0,:] = dload*np.random.randn(len(Gmacro[0,:]))
    
    for i in range(Nsteps-1):
        Gmacro[i+1,:] = Gmacro[i,:] + dload*np.random.randn(len(Gmacro[0,:]))

    start = timer()  
    micromodel.restart_initial_guess()
    for i in range(Nsteps):
        # micromodel.setUpdateFlag(False)
        micromodel.solve_microproblem(Gmacro[i,:])
        A_ref = micromodel.homogenise_tangent()
    end = timer()
    time_ref += (end-start)
        
    start = timer()  
    micromodel.restart_initial_guess()
    for i in range(Nsteps):
        # micromodel.setUpdateFlag(False)
        A_DF = get_tangent_pertubation_forward(Gmacro[i,:], micromodel, tau)
    end = timer()
    time_DF += (end-start)
        
    start = timer()  
    micromodel.restart_initial_guess()
    for i in range(Nsteps):
        # micromodel.setUpdateFlag(False)
        A_CD = get_tangent_pertubation_central_difference(Gmacro[i,:], micromodel, tau)
    end = timer()
    time_CD += (end-start)


    print("time ref", time_ref)
    print("speedup CD:" , time_CD/time_ref)
    print("speedup DF:" , time_DF/time_ref)