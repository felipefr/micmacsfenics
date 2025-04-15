"""
Created on Wed Mar 24 12:47:54 2021

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

import sys
import multiphenics as mp
import dolfin as df
import fetricks as ft

class MultiscaleFormulation:

    def __init__(self, mesh, others):

        self.mesh = mesh
        self.others = others

        V = self.flutuationSpace()
        R = self.zeroAverageSpace()
        restrictions = [None, None] + self.otherRestrictions()
        W = [V, R] + self.otherSpaces()
        self.W = mp.BlockFunctionSpace(W, restrict=restrictions)

        self.uu = mp.BlockTrialFunction(self.W)
        self.vv = mp.BlockTestFunction(self.W)
        self.uu_ = mp.block_split(self.uu)
        self.vv_ = mp.block_split(self.vv)

    def __call__(self, C, Eps):
        return self.blocks(C, Eps) + self.bcs() + [self.W]

    def blocks(self, C, Eps):
        dx = df.Measure('dx', self.mesh)

        u, p = self.uu_[0:2]
        v, q = self.vv_[0:2]

        aa = [[df.inner(df.dot(C, ft.symgrad_mandel(u)), ft.symgrad_mandel(v))*dx, df.inner(p, v)*dx],
              [df.inner(q, u)*dx(), 0]]

        # dot(sigma(Eps) , symgrad(v)) = dot(Eps , sigma(symgrad(v))
        ff = [-df.inner(df.dot(C,Eps), ft.symgrad_mandel(v))*dx, 0]

        return [aa, ff]

    def bcs(self):
        return [[]]

    def enrichBlocks(self):
        pass

    def createMixedSpace(self):
        pass

    def flutuationSpace(self):
        return df.VectorFunctionSpace(self.mesh, "CG",
                                      self.others['polyorder'])

    def zeroAverageSpace(self):
        return df.VectorFunctionSpace(self.mesh, "Real", 0)

    def otherSpaces(self):
        return []

    def otherRestrictions(self):
        return []
