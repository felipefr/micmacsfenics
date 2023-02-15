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

from material_model_expression import materialModelExpression
from micmacsfenics.core.fenicsUtils import (symgrad, tensor2mandel,  mandel2tensor, tr_mandel, Id_mandel_np)

# Constant materials params
class hyperlasticityModelExpression(materialModelExpression):
    
    def __init__(self, W, dxm, param):
        self.lamb = param['lamb']
        self.mu = param['mu']
        self.alpha = param['alpha']
        
        super().__init__(W, dxm)
    
    def stressHomogenisation(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format
    
        ee = np.dot(e,e)
        tre2 = (e[0] + e[1])**2.0
        
        lamb_star = self.lamb*( 1 + self.alpha*tre2)
        mu_star = self.mu*( 1 + self.alpha*ee ) 
        
        return lamb_star*(e[0] + e[1])*Id_mandel_np + 2*mu_star*e
    
    
    def tangentHomogenisation(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format
        
        ee = np.dot(e,e)
        tre2 = (e[0] + e[1])**2.0
        
        lamb_star = self.lamb*( 1 + 3*self.alpha*tre2)
        mu_star = self.mu*( 1 + self.alpha*ee ) 
        
        D = 4*self.mu*self.alpha*np.outer(e,e)
    
        D[0,0] += lamb_star + 2*mu_star
        D[1,1] += lamb_star + 2*mu_star
        D[0,1] += lamb_star
        D[1,0] += lamb_star
        D[2,2] += 2*mu_star

        return D