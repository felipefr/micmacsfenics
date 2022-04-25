#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:48:27 2022

@author: felipe
"""

import sys
import dolfin as df
import numpy as np

sys.path.insert(0, '../../core/')

from micmacsfenics.core.fenicsUtils import LocalProjector

from generic_gausspoint_expression import genericGaussPointExpression
    
class materialModelExpression:
    def __init__(self, W, dxm):
        self.strain = df.Function(W) 
        self.projector = LocalProjector(W, dxm)
        
        self.stress = genericGaussPointExpression(self.strain, self.stressHomogenisation , (3,))
        self.tangent = genericGaussPointExpression(self.strain, self.tangentHomogenisation , (3,3,))
        
    def stressHomogenisation(self, e, cell = None):
        pass
    
    def tangentHomogenisation(self,e, cell = None):
        pass
    
    def updateStrain(self, e):
        self.projector(e, self.strain)
