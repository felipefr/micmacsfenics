#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:18:10 2021

@author: felipefr
"""
import dolfin as df
import numpy as np


halfsqrt2 = 0.5*np.sqrt(2)
Id_mandel_df = df.as_vector([1.0, 1.0, 0.0])
Id_mandel_np = np.array([1.0, 1.0, 0.0])

def mandel2tensor_np(X):
    return np.array([[X[0], halfsqrt2*X[2]],
                     [halfsqrt2*X[2], X[1]]])

def tensor2mandel_np(X):
    return np.array([X[0,0], X[1,1], halfsqrt2*(X[0,1] + X[1,0])])


def tensor2mandel(X):
    return df.as_vector([X[0,0], X[1,1], halfsqrt2*(X[0,1] + X[1,0])])
                      
def mandel2tensor(X):
    return df.as_tensor([[X[0], halfsqrt2*X[2]],
                        [halfsqrt2*X[2], X[1]]])
      
def tr_mandel(X):
    return X[0] + X[1]


def symgrad(v): return df.sym(df.nabla_grad(v))

def symgrad_voigt(v):
    return df.as_vector([v[0].dx(0), v[1].dx(1), v[0].dx(1) + v[1].dx(0)])

def macro_strain(i): ## this is voigt
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return np.array([[Eps_Voigt[0], Eps_Voigt[2]/2.],
                    [Eps_Voigt[2]/2., Eps_Voigt[1]]])

def macro_strain_mandel(i): 
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return np.array([[Eps_Voigt[0], halfsqrt2*Eps_Voigt[2]],
                    [halfsqrt2*Eps_Voigt[2], Eps_Voigt[1]]])

def stress2Voigt(s):
    return df.as_vector([s[0, 0], s[1, 1], s[0, 1]])


def strain2Voigt(e):
    return df.as_vector([e[0, 0], e[1, 1], 2*e[0, 1]])

def voigt2strain(e):
    return df.as_tensor([[e[0], 0.5*e[2]], [0.5*e[2], e[1]]])


def voigt2stress(s):
    return df.as_tensor([[e[0], e[2]], [e[2], e[1]]])

def Integral(u, dx, shape):
    n = len(shape)
    valueIntegral = np.zeros(shape)

    if(n == 1):
        for i in range(shape[0]):
            valueIntegral[i] = df.assemble(u[i]*dx)

    elif(n == 2):
        for i in range(shape[0]):
            for j in range(shape[1]):
                valueIntegral[i, j] = df.assemble(u[i, j]*dx)

    return valueIntegral


class LocalProjector:
    def __init__(self, V, dx):    
        self.dofmap = V.dofmap()
        
        dv = df.TrialFunction(V)
        v_ = df.TestFunction(V)
        
        a_proj = df.inner(dv, v_)*dx
        self.b_proj = lambda u: df.inner(u, v_)*dx
        
        self.solver = df.LocalSolver(a_proj)
        self.solver.factorize()
        
        self.sol = df.Function(V)
    
    def __call__(self, u, sol = None):
        b = df.assemble(self.b_proj(u))
        
        if sol is None:
            self.solver.solve_local(self.sol.vector(), b,  self.dofmap)
            return self.sol
        else:
            self.solver.solve_local(sol.vector(), b,  self.dofmap)
            return
    
# def local_project(v, V, dxm, u=None):
#     dv = df.TrialFunction(V)
#     v_ = df.TestFunction(V)
#     a_proj = df.inner(dv, v_)*dxm
#     b_proj = df.inner(v, v_)*dxm

#     solver = df.LocalSolver(a_proj)
#     solver.factorize()
    
#     b = df.assemble(b_proj)
    
#     if u is None:
#         u = df.Function(V)
#         solver.solve_local(u.vector(), b,  V.dofmap())
#         # solver.solve_local_rhs(u)
#         return u
#     else:
#         solver.solve_local(u.vector(), b,  V.dofmap())
#         return