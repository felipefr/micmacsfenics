#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:59:15 2022

@author: felipe
"""

import dolfin as df 
import numpy as np
import ufl
from ddfenics.fenics.fenicsUtils import local_project

typesForProjection = (ufl.tensors.ComponentTensor, ufl.tensors.ListTensor, 
                      ufl.algebra.Sum, ufl.tensoralgebra.Dot, ufl.differentiation.Grad)

# Class to convert raw data to fenics tensorial function objects (and vice-versa)    
class DDFunction(df.Function):
    
    def __init__(self, V):
        super().__init__(V)  
        
        self.V = self.function_space()        
        self.n = self.V.num_sub_spaces()
                
        self.metadata = {}   
        if(self.V.ufl_element().family() == 'Quadrature'):
            self.metadata = {"quadrature_degree": self.V.ufl_element().degree(), "quadrature_scheme": "default"}
        
        self.d = self.__getBlankData()

        self.MAP = self.__createMapping()

    def data(self):        
        for i in range(self.n):
            self.d[self.map[i,0,:], i] = self.vector().get_local()[self.map[i,1,:]]
            
        return self.d
    
    
    # def data2(self):
    #     return self.vector().get_local()[self.MAP]  
    
    def update(self, d): # Maybe improve perfomance
        
        if isinstance(d, np.ndarray):
            self.vector().set_local(self.__data2tensorVec(d, self.vector().get_local()[:]))
            
        elif isinstance(d, typesForProjection):
            self.assign(local_project(d, self.V, metadata = self.metadata))
        
        elif isinstance(d, df.Function) and d.function_space() == self.V:
            self.assign(d)
            
        else:
            print("DDFunction.update: Invalid type")
            print(type(d))
            input()
            
            
    def __data2tensorVec(self, d, tenVec = None):
        if(type(tenVec) == type(None)):
            tenVec = np.zeros(self.dim())
            
        for i in range(self.n):
            tenVec[self.map[i,1,:]] = d[self.map[i,0,:], i]
        
        return tenVec
    
    def __createMapping(self): 

        self.map = []
        for i in range(self.n):
            mapTemp = self.V.sub(i).collapse(True)[1]
            self.map.append( [np.array(list(x)) for x in [mapTemp.keys(), mapTemp.values()]])   
            
        self.map = np.array(self.map)

        MAP = np.zeros((self.map.shape[2], self.n), dtype = int)        
        for i in range(self.n):
            MAP[self.map[i,0,:], i] = self.map[i,1,:]
        
        return MAP
        
        
    def __getBlankData(self):
        return np.zeros(self.V.dim()).reshape((-1, self.n)) 
