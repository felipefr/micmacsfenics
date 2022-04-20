#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:48:27 2022

@author: felipe
"""

import dolfin as df

class genericHomogenisedExpression(df.UserExpression):
    def __init__(self, strain, homogenisationLaw, shape,  **kwargs):
        self.strain = strain
        self.homogenisationLaw = homogenisationLaw
        self.shape = shape
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        strain = self.strain.vector().get_local()[cell.index*3:(cell.index + 1)*3]
        values[:] = self.homogenisationLaw(strain).flatten()

    def value_shape(self):
        return self.shape
    
class homogenisedModel:
    def __init__(self, strain):
        self.strain = strain
        
        self.stress = genericHomogenisedExpression(self.strain, self.stressHomogenisation , (3,))
        self.tangent = genericHomogenisedExpression(self.strain, self.tangentHomogenisation , (3,3,))
        
    def stressHomogenisation(self, e):
        pass
    
    def tangentHomogenisation(self,e):
        pass

class homogenisedHyperlasticityModel(homogenisedModel):
    
    def __init__(self, strain, lamb, mu, alpha):
        self.lamb = lamb
        self.mu = mu
        self.alpha = alpha
        
        super().__init__(strain)
    
    def stressHomogenisation(self, e): # elastic (I dont know why for the moment) # in mandel format
    
        ee = np.dot(e,e)
        tre2 = (e[0] + e[1])**2.0
        
        lamb_star = self.lamb*( 1 + self.alpha*tre2)
        mu_star = self.mu*( 1 + self.alpha*ee ) 
        
        return lamb_star*(e[0] + e[1])*Id_mandel_np + 2*mu_star*e
    
    
    def tangentHomogenisation(self, e): # elastic (I dont know why for the moment) # in mandel format
        
        ee = np.dot(e,e)
        tre2 = (e[0] + e[1])**2.0
        
        lamb_star = self.lamb*( 1 + 3*self.alpha_*tre2)
        mu_star = self.mu*( 1 + self.alpha*ee ) 
        
        D = 4*self.mu*self.alpha*np.outer(e,e)
    
        D[0,0] += lamb_star + 2*mu_star
        D[1,1] += lamb_star + 2*mu_star
        D[0,1] += lamb_star
        D[1,0] += lamb_star
        D[2,2] += 2*mu_star
        
        return D