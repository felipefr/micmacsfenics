#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 13:11:17 2022

@author: ffiguere
"""

import sys
import numpy as np
#import multiphenics as mp
import dolfin as df
from timeit import default_timer as timer
from ufl import nabla_div, indices
from functools import partial 

from fetricks.mechanics.conversions3d import * 
from fetricks.mechanics.elasticity_conversions import youngPoisson2lame

class MicroConstitutiveModelFake: # Make it a abstract class (make it simpler)

    # Counter of calls 
    countComputeFluctuations = 0
    countComputeCanonicalProblem = 0
    countTangentCalls = 0
    countStressCalls = 0
   
    def __init__(self, param, ndim = 2, tensor_encoding = 'mandel'):
        nu, lamb = param
        self.param = param
        self.lame = youngPoisson2lame(nu, lamb)
        self.tensor_encoding = "mandel"
        self.ndim = ndim
        self.strain_dim = int(self.ndim*(self.ndim + 1)/2)
        self.tensor_encoding = tensor_encoding        
        
        self.isStressUpdated = False
        self.isTangentUpdated = False
        self.isFluctuationUpdated = False
        
        # self.getStress = self.__computeStress
        self.getTangent = self.__computeTangent
        
        self.stresshom = np.zeros(self.strain_dim)
        self.tangenthom = np.zeros((self.strain_dim,self.strain_dim))

    def setUpdateFlag(self, flag):
        self.setStressUpdateFlag(flag)
        self.setTangentUpdateFlag(flag)
        self.setFluctuationUpdateFlag(flag)
        
    def setStressUpdateFlag(self, flag):
        pass
        # if(flag):
            # self.getStress = self.__returnStress
        # else:
            # self.getStress = self.__computeStress

    def getStress(self,e):
        self.stresshom = self.lame[0]*tr_mandel(e)*Id_mandel_np + 2*self.lame[1]*e
        return self.stresshom

    def setTangentUpdateFlag(self, flag):
        if(flag):
            self.getTangent = self.__returnTangent 
        else:
            self.getTangent = self.__computeTangent
    
    def setFluctuationUpdateFlag(self, flag):
        self.isFluctuationUpdated = flag
        
    def getStressTangent(self, e):
        # return np.concatenate((self.getStress(e), symflatten(self.getTangent(e))))
        return self.getStress(e), symflatten(self.getTangent(e))

    def __returnTangent(self, e):
        self.eps = e
        type(self).countTangentCalls = type(self).countTangentCalls + 1     
        return self.tangenthom
    
    def __returnStress(self, e):
        self.eps = e
        type(self).countStressCalls = type(self).countStressCalls + 1     
        return self.stresshom

    def __homogeniseTangent(self):
        type(self).countComputeCanonicalProblem = type(self).countComputeCanonicalProblem + 1
            
    def __homogeniseStress(self):
        self.stresshom = self.lame[0]*tr_mandel(self.eps)*Id_mandel_np + 2*self.lame[1]*self.eps   
        
    def __computeFluctuations(self, e):
        self.eps = e
        self.setFluctuationUpdateFlag(True)
        type(self).countComputeFluctuations = type(self).countComputeFluctuations + 1

    def __computeStress(self, e):
        self.eps = e
        if(not self.isFluctuationUpdated):
            self.__computeFluctuations(e)
    
        self.__homogeniseStress()
        self.setStressUpdateFlag(True)
    
        return self.__returnStress(e)
        
    def __computeTangent(self, e):
        self.eps = e
        
        if(not self.isFluctuationUpdated):
            self.__computeFluctuations(e)
            
        self.__homogeniseTangent()
        self.setTangentUpdateFlag(True)            
        
        return self.__returnTangent(e)
