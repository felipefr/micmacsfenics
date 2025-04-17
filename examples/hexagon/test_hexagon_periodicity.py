#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 19:16:29 2025

@author: frocha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:18:21 2025

@author: felipe
"""

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
from fetricks.mechanics.elasticity_conversions import youngPoisson2lame
from fetricks.mechanics.elasticity_conversions import youngPoisson2lame_planeStress
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
    E = 20e3
    nu = 0.3
    
    lamb, mu = youngPoisson2lame(E,nu) # plane strain
    # lamb, mu = youngPoisson2lame_planeStress(E,nu)
    bndModel_micro = ['MRHOHP', [1,2,3]]
    # bndModel_micro = ['hexper', [1,2,3]]
    
    print(lamb + 2*mu)
    print(mu)
    
    print(lamb, mu)
    psi_mu = ft.psi_hookean_nonlinear_lame
    mesh_micro = ft.Mesh(mesh_micro_name)
    
    param_micro = {"lamb" : df.Constant(lamb) , 
                   "mu": df.Constant(mu), 
                   "alpha": df.Constant(0.0)} 
    
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
    mesh_micro_name_geo = './meshes/mesh_hexagon.geo'.format(gdim)
    mesh_micro_name = './meshes/mesh_hexagon.xdmf'.format(gdim)
    
    # mesh_micro_name_geo = './meshes/mesh_square.geo'.format(gdim)
    # mesh_micro_name = './meshes/mesh_square.xdmf'.format(gdim)
    
    gmsh_mesh = ft.GmshIO(mesh_micro_name_geo, gdim)
    gmsh_mesh.write("xdmf")
        
    micromodel = getMicroModel(mesh_micro_name, gdim)  
    
    micromodel.setCanonicalproblem()
    micromodel.restart_initial_guess()
    micromodel.setUpdateFlag(False)
    
    eta = 8
    a = 1
    t = a/eta 
    Arve = 0.5*np.sqrt(3)*a**2
    As = Arve - 0.5*np.sqrt(3)*(a-t)**2
    fac = As/Arve

    # KGG = micromodel.compute_tangent_multiphenics()
    KGG, SG = micromodel.compute_tangent_localisation_tensors()
    # KHH, SH = micromodel.compute_hypertangent()
    # KGH = micromodel.compute_mixedtangent(SG,SH)
    
    KGG = fac*KGG
    mu1 = KGG[2,2]/2.0
    lamb1 = KGG[0,0] - 2*mu1
    lamb2 = KGG[0,1]
    mu2 = (KGG[0,0]-lamb2)/2
    
    np.set_printoptions(precision=3)
    print('KGG=', KGG)
    print(mu1,mu2)
    print(lamb1,lamb2)
    print('cP [GPa]=', (lamb1 + 2*mu1)*10**(-3))
    print('cS [GPa]=', mu2*10**(-3))

    # print('KHH=', fac*KHH)
    # print('KGH=', fac*KGH)