#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul  6 20:23:09 2024

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or <felipe.f.rocha@gmail.com>
"""

import numpy as np
from dolfinx import fem
import ufl

import fetricksx as ft
    
class MicroMacro:
    
    def __init__(self, W, Wtan, dxm, micromodels=None):
        
        self.mesh = W.mesh
        self.W = W
        self.Wtan = Wtan
        self.dxm = dxm 
        self.size_tan = self.Wtan.space.num_sub_spaces
        self.size_strain = self.W.space.num_sub_spaces
        self.stress = fem.Function(self.W.space)
        self.strain = fem.Function(self.W.space)
        self.tangent = fem.Function(self.Wtan.space)

        self.strain_array = self.strain.x.array.reshape( (self.W.nq_mesh, self.size_strain) )
        self.stress_array = self.stress.x.array.reshape( (self.W.nq_mesh, self.size_strain))
        self.tangent_array = self.tangent.x.array.reshape( (self.W.nq_mesh, self.size_tan))
        
        if(micromodels):
            self.micromodels = micromodels
        else:
            self.micromodels = self.W.nqpts*[None] # just a placeholder 
        
    def set_track_strain(self, strain):
        self.strain_evaluator = ft.QuadratureEvaluator(strain, self.strain_array, self.mesh, self.W)
        
    # afterward micromodel setting
    def set_micromodel(self, micro_model, i):
        self.micromodels[i] = micro_model
                
    def tangent_op(self, de):
        return ufl.dot(ft.as_sym_tensor_3x3(self.tangent), de) 

    def update(self, dummy1, dummy2): # dummy variables to fit callback arguments
        self.strain_evaluator()
        
        for s, t, e, m in zip(self.stress_array, self.tangent_array, self.strain_array, self.micromodels):
            m.solve_microproblem(e) 
            s[:], t[:] = m.get_stress_tangent()  


