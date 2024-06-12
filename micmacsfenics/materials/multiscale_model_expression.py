#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:50:48 2022

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""
import dolfin as df
import numpy as np

import micmacsfenics as mm 

class MultiscaleModelExpression(mm.MaterialModelExpression):
    
    def __init__(self, W, dxm, micromodels):
        self.micromodels = micromodels
        super().__init__(W, dxm)
    
    def stressHomogenisation(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format        
        return self.micromodels[cell.index].getStress(e)
    
    def tangentHomogenisation(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format
        
        return self.micromodels[cell.index].getTangent(e)
    
    def updateStrain(self, e):
        super().updateStrain(e)
        
        for m in self.micromodels:
            m.setUpdateFlag(False)
    