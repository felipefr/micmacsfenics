#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:18:10 2021

@author: felipefr
"""
import dolfin as df
import numpy as np


# symgrad = lambda v: df.sym(df.nabla_grad(v))
symgrad = lambda v: 0.5*(df.grad(v) + df.grad(v).T)
symgrad_voigt = lambda v: df.as_vector([v[0].dx(0), v[1].dx(1), v[0].dx(1) + v[1].dx(0) ])

# def Integral(u,dx,shape):
    
#     n = len(shape)
#     I = np.zeros(shape)
    
#     if(type(dx) != type([])):
#         dx = [dx]
 
#     if(n == 1):
#         for i in range(shape[0]):
#             for dxj in dx:
#                 I[i] += df.assemble(u[i]*dxj)
            
#     elif(n == 2):
#         for i in range(shape[0]):
#             for j in range(shape[1]):
#                 for dxk in dx:
#                     I[i,j] += df.assemble(u[i,j]*dxk)
    
#     else:
#         print('not implement for higher order integral')
        
    
#     return I


def Integral(u,dx,shape):
    
    n = len(shape)
    I = np.zeros(shape)
 
    if(n == 1):
        for i in range(shape[0]):
            I[i] = df.assemble(u[i]*dx)
            
    elif(n == 2):
        for i in range(shape[0]):
            for j in range(shape[1]):
                I[i,j] = df.assemble(u[i,j]*dx)

    
    return I