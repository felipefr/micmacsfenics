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
