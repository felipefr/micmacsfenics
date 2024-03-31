#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 20:52:20 2023

@author: ffiguere


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

# NOT TESTED

import numpy as np
import dolfin as df
import fetricks as ft

class multiscaleModelExpression(ft.materialModelExpression):
    
    def __init__(self, W, Wtan, dxm, micromodels):
        self.micromodels = micromodels
        super().__init__(W, Wtan, dxm)
    
    def pointwise_stress(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format        
        return self.micromodels[cell.index].getStress(e)
    
    def pointwise_tangent(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format
        return self.micromodels[cell.index].getTangent(e).flatten()[ft.ind_sym_tensor_3x3]
    
    def tangent_op(self, de):
        return df.dot(ft.as_sym_tensor_3x3(self.tangent), de) 
    
    def update(self, e):
        super().update(e)
        
        for m in self.micromodels:
            m.setUpdateFlag(False)
    
