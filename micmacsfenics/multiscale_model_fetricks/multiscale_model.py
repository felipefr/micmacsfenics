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

import sys
import dolfin as df
import numpy as np
import fetricks as ft

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF

class multiscaleModel(ft.materialModel):
    
    def __init__(self, W, Wtan, dxm, micromodels):
        
        self.micromodels = micromodels
        self.mesh = W.mesh()
        self.__createInternalVariables(W, Wtan, dxm)

    def __createInternalVariables(self, W, Wtan, dxm):
        self.stress = df.Function(W)
        self.strain = df.Function(W)
        self.tangent = df.Function(Wtan)
        
        self.size_tan = Wtan.num_sub_spaces()
        self.size_strain = W.num_sub_spaces()
        self.ngauss = int(W.dim()/self.size_strain)
        
        self.projector_strain = ft.LocalProjector(W, dxm, sol = self.strain)
                
        self.strain_array = self.strain.vector().vec().array.reshape( (self.ngauss, self.size_strain) )
        self.stress_array = self.stress.vector().vec().array.reshape( (self.ngauss, self.size_strain))
        self.tangent_array = self.tangent.vector().vec().array.reshape( (self.ngauss, self.size_tan))
                
    def tangent_op(self, de):
        return df.dot(ft.as_sym_tensor_3x3(self.tangent), de) 

    def update_stress_tangent(self):
        for s, t, e, m in zip(self.stress_array, self.tangent_array, self.strain_array, self.micromodels):
            s[:], t[:] = m.getStressTangent_force(e)  
            
    def update(self, strain_new):
        self.projector_strain(strain_new) 
        self.update_stress_tangent()    

    