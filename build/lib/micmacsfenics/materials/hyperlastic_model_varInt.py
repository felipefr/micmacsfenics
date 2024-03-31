#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:15:36 2022

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

import sys
import dolfin as df
from material_model import materialModel 


sys.path.insert(0, '../../core/')

from fenicsUtils import symgrad, tensor2mandel,  mandel2tensor, tr_mandel, Id_mandel_df, Id_mandel_np

class hyperlasticityModel_varInt(materialModel):
    
    def __init__(self,E,nu, alpha):
        self.lamb = df.Constant(E*nu/(1+nu)/(1-2*nu))
        self.mu = df.Constant(E/2./(1+nu))
        self.alpha = df.Constant(alpha)
        
    def createInternalVariables(self, W, W0):
        self.sig = df.Function(W)
        self.eps = df.Function(W)
        self.tre2 = df.Function(W0)
        self.ee = df.Function(W0)
    
        self.varInt = {'tre2': self.tre2, 'ee' : self.ee, 'eps' : self.eps,  'sig' : self.sig} 

    def sigma(self, lamb_, mu_, eps): # elastic (I dont know why for the moment) # in mandel format
        return lamb_*tr_mandel(eps)*Id_mandel_df + 2*mu_*eps
    
    def epssymgrade(self, de):
        return df.inner(self.eps, de)*self.eps

    
    def tangent(self, de):
        lamb_ = self.lamb*( 1 + 3*self.alpha*self.tre2)
        mu_ = self.mu*( 1 + self.alpha*self.ee ) 
        
        de_mandel = tensor2mandel(de)
        
        return self.sigma(lamb_, mu_, de_mandel)  + 4*self.mu*self.alpha*self.epssymgrade(de_mandel)

    def update_alpha(self, symgradnew):
        
        ee = df.inner(symgradnew,symgradnew)
        tre2 = tr_mandel(symgradnew)**2.0
        
        lamb_ = self.lamb*( 1 + self.alpha*tre2)
        mu_ = self.mu*( 1 + self.alpha*ee ) 
        
        alpha_new = {'tre2': tre2, 'ee' : ee, 'eps' : symgradnew, 'sig': self.sigma(lamb_, mu_, symgradnew)}
        self.project_var(alpha_new)
