#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 00:38:37 2024

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 00:23:16 2022

@author: felipe


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""


import os, sys
import matplotlib.pyplot as plt

import numpy as np
from dolfinx import fem, io
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from mpi4py import MPI

import fetricksx as ft

class MicroMacro:
    
    def __init__(self, W, Wtan, dxm, micromodels=None):
        
        self.mesh = W.mesh()
        self.create_internal_variables(W, Wtan, dxm)
        
        if(micromodels):
            self.micromodels = micromodels
        else:
            self.micromodels = self.ngauss*[None] # just a placeholder 
        
    
    # afterward micromodel setting
    def set_micromodel(self, micro_model, i):
        self.micromodels[i] = micro_model
        
    def create_internal_variables(self, W, Wtan, dxm):
        self.stress = fem.Function(W)
        self.strain = fem.Function(W)
        self.tangent = fem.Function(Wtan)
        
        self.size_tan = Wtan.num_sub_spaces()
        self.size_strain = W.num_sub_spaces()
        self.ngauss = int(W.dim()/self.size_strain)
        
        self.projector_strain = ft.LocalProjector(W, dxm, sol = self.strain)
                
        self.strain_array = self.strain.vector().vec().array.reshape( (self.ngauss, self.size_strain) )
        self.stress_array = self.stress.vector().vec().array.reshape( (self.ngauss, self.size_strain))
        self.tangent_array = self.tangent.vector().vec().array.reshape( (self.ngauss, self.size_tan))
                
    def tangent_op(self, de):
        return ufl.dot(ft.as_sym_tensor_3x3(self.tangent), de) 

 # is forced? force to update state even it is already updated
    def update_stress_tangent(self, is_forced = True):
        
        if(is_forced):
            for s, t, e, m in zip(self.stress_array, self.tangent_array, self.strain_array, self.micromodels):
                s[:], t[:] = m.getStressTangent_force(e)  
        else:
            for s, t, e, m in zip(self.stress_array, self.tangent_array, self.strain_array, self.micromodels):
                s[:], t[:] = m.getStressTangent(e)  

    def update(self, strain_new, is_forced = True):
        self.projector_strain(strain_new) 
        self.update_stress_tangent(is_forced)    

    def stress_op(self, e):
        pass
    
    def param_parser(self, param):
        pass
