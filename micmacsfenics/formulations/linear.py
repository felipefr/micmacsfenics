#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:56:26 2021

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""
import multiphenics as mp
import dolfin as df
from micmacsfenics.formulations.multiscale_formulation import MultiscaleFormulation


class FormulationLinear(MultiscaleFormulation):
    def bcs(self):

        uD = self.others['uD'] if 'uD' in self.others else df.Constant((0, 0))
        
        if 'external_bnd' in self.others:
            bcs = [ mp.DirichletBC(self.W.sub(0), uD, self.mesh.boundaries, i) 
                     for i in self.others['external_bnd'] ]
        else:
            onBoundary = df.CompiledSubDomain('on_boundary')
            bcs = [mp.DirichletBC(self.W.sub(0), uD, onBoundary)]
            
        return [mp.BlockDirichletBC(bcs)]
