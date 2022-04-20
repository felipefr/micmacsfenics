#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:18:46 2022

@author: felipe
"""

import sys
import dolfin as df
import numpy as np
from material_model import materialModel 


sys.path.insert(0, '../../core/')

from fenicsUtils import (symgrad, tensor2mandel,  mandel2tensor, tr_mandel, Id_mandel_df,
                        Id_mandel_np, LocalProjector)



class hyperlasticityModel(materialModel):
    
    def __init__(self, param):
        
        if('lamb' in param.keys()):
            self.lamb_ = param['lamb']
            self.mu_ = param['mu']
            
        else: 
            E = param['E']
            nu = param['nu']
            self.lamb_ = E*nu/(1+nu)/(1-2*nu)
            self.mu_ = E/2./(1+nu)

        self.alpha_ = param['alpha']  if 'alpha' in param.keys() else 0.0
        
        self.lamb = df.Constant(self.lamb_)
        self.mu = df.Constant(self.mu_)
        self.alpha = df.Constant(self.alpha_)
        
        
        
    def createInternalVariables(self, W, W0, dxm):
        self.sig = df.Function(W)
        self.eps = df.Function(W)
        
        projector = LocalProjector(W, dxm)
        
        self.varInt = {'eps' : self.eps,  'sig' : self.sig}
        self.projector_list = {'eps' : projector,  'sig' : projector}

    def sigma(self, lamb_, mu_, eps): # elastic (I dont know why for the moment) # in mandel format
        return lamb_*tr_mandel(eps)*Id_mandel_df + 2*mu_*eps
    
    def epseps_de(self, de):
        return df.inner(self.eps, de)*self.eps
    
    def tangent(self, de):
        ee = df.inner(self.eps, self.eps)
        tre2 = tr_mandel(self.eps)**2.0
        
        lamb_ = self.lamb*( 1 + 3*self.alpha*tre2)
        mu_ = self.mu*( 1 + self.alpha*ee ) 
        
        de_mandel = tensor2mandel(de)
        
        return self.sigma(lamb_, mu_, de_mandel)  + 4*self.mu*self.alpha*self.epseps_de(de_mandel)

    def update_alpha(self, epsnew):
        
        ee = df.inner(epsnew, epsnew)
        tre2 = tr_mandel(epsnew)**2.0
        
        lamb_ = self.lamb*( 1 + self.alpha*tre2)
        mu_ = self.mu*( 1 + self.alpha*ee ) 
        
        alpha_new = {'eps' : epsnew, 'sig': self.sigma(lamb_, mu_, epsnew)}
        self.project_var(alpha_new)
        
        
    def stress_np(self, e): # elastic (I dont know why for the moment) # in mandel format
    
        ee = np.dot(e,e)
        tre2 = (e[0] + e[1])**2.0
        
        lamb_star = self.lamb_*( 1 + self.alpha_*tre2)
        mu_star = self.mu_*( 1 + self.alpha_*ee ) 
        
        return lamb_star*(e[0] + e[1])*Id_mandel_np + 2*mu_star*e
    
    
    def tangent_np(self, e): # elastic (I dont know why for the moment) # in mandel format
        
        ee = np.dot(e,e)
        tre2 = (e[0] + e[1])**2.0
        
        lamb_star = self.lamb_*( 1 + 3*self.alpha_*tre2)
        mu_star = self.mu*( 1 + self.alpha_*ee ) 
        
        D = 4*self.mu_*self.alpha_*np.outer(e,e)
    
        D[0,0] += lamb_star + 2*mu_star
        D[1,1] += lamb_star + 2*mu_star
        D[0,1] += lamb_star
        D[1,0] += lamb_star
        D[2,2] += 2*mu_star
        
        return D