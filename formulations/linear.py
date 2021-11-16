#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:56:26 2021

@author: felipefr
"""
import multiphenics as mp
import dolfin as df
from micmacsfenics.formulations.multiscale_formulation import MultiscaleFormulation


class FormulationLinear(MultiscaleFormulation):
    def bcs(self):
        onBoundary = df.CompiledSubDomain('on_boundary')
        uD = self.others['uD'] if 'uD' in self.others else df.Constant((0, 0))

        bc1 = mp.DirichletBC(self.W.sub(0), uD, onBoundary)
        return [mp.BlockDirichletBC([bc1])]
