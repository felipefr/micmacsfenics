#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:09:55 2022

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""



import sys

sys.path.insert(0, '../../core/')


class materialModel:
    
    def stress_op(self, e):
        pass
    
    def tangent_op(self, de):
        pass
    
    def createInternalVariables(self, W, W0, dxm):
        pass
    
    def update(self,deps, old_sig, old_p):
        pass
    
    def project_var(self, AA):
        for label in AA.keys(): 
            self.projector_list[label](AA[label], self.varInt[label])