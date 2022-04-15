#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:18:10 2021

@author: felipefr
"""
import dolfin as df
import numpy as np


def symgrad(v): return df.sym(df.nabla_grad(v))


def symgrad_voigt(v):
    return df.as_vector([v[0].dx(0), v[1].dx(1), v[0].dx(1) + v[1].dx(0)])


def macro_strain(i):
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return np.array([[Eps_Voigt[0], Eps_Voigt[2]/2.],
                    [Eps_Voigt[2]/2., Eps_Voigt[1]]])


def stress2Voigt(s):
    return df.as_vector([s[0, 0], s[1, 1], s[0, 1]])


def strain2Voigt(e):
    return df.as_vector([e[0, 0], e[1, 1], 2*e[0, 1]])

def voigt2strain(e):
    return df.as_tensor([[e[0], 0.5*e[2]], [0.5*e[2], e[1]]])

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
