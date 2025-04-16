#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 15:44:18 2025

@author: frocha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:57:46 2021

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

import dolfin as df
import numpy as np
from micmacsfenics.formulations.multiscale_formulation import MultiscaleFormulation


class HexagonalPeriodicBoundary(df.SubDomain):
    # Left boundary is "target domain" G
    def __init__(self, a = 1.0, phase = -np.pi, **kwargs):
        self.a = a # distance between two periodic faces
        self.angles = np.linspace(phase, phase + 2*np.pi/3., 3)
        # unit vector periodicities
        self.v = np.array([[np.cos(ang), np.sin(ang)] for ang in self.angles])
        
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        # return True if on face 0,1 or 2, excluding corners 0-5 and 2-3
        if(on_boundary):
            return ( self.is_face1(x) or
                     (self.is_face0(x) and not self.is_face5(x)) or 
                     (self.is_face2(x) and not self.is_face3(x)) )

        return False

    # clock-wise rotation of theta = ang
    def rotation_matrix(self, ang):
        return np.array([[np.cos(ang),np.sin(ang)],
                         [-np.sin(ang), np.cos(ang)]])
    
    def is_on_face_i(self, x, i, is_opposite = 0.0):
        y = self.rotation_matrix(self.angles[i] + is_opposite*np.pi)@x
        return df.near(y[0], 0.5*self.a)
    
    def is_face0(self, x):
        return self.is_on_face_i(x, 0)

    def is_face1(self, x):
        return self.is_on_face_i(x, 1)

    def is_face2(self, x):
        return self.is_on_face_i(x, 2)

    def is_face3(self, x):
        return self.is_on_face_i(x, 0, is_opposite=1.0)
    
    def is_face4(self, x):
        return self.is_on_face_i(x, 1, is_opposite=1.0)

    def is_face5(self, x):
        return self.is_on_face_i(x, 2, is_opposite=1.0)
 
    def map(self, x, y):
        if((self.is_face3(x) and self.is_face4(x)) or (self.is_face4(x) and self.is_face5(x)) ):
            y[:] = x[:] + self.a*self.v[1,:] 
        elif(self.is_face0(x) and self.is_face5(x)):
            y[:] = x[:] + self.a*self.v[2,:]
        elif(self.is_face2(x) and self.is_face3(x)):
            y[:] = x[:] + self.a*self.v[0,:]
        elif(self.is_face3(x)):
            y[:] = x[:] + self.a*self.v[0,:]
        elif(self.is_face4(x)):
            y[:] = x[:] + self.a*self.v[1,:]
        elif(self.is_face5(x)):
            y[:] = x[:] + self.a*self.v[2,:]
        else:
            y[0] = x[0]
            y[1] = x[1]
        

class FormulationHexagonalPeriodic(MultiscaleFormulation):

    def flutuationSpace(self):
        polyorder = self.others['polyorder']
        self.periodicity = HexagonalPeriodicBoundary()

        return df.VectorFunctionSpace(self.mesh, "CG", polyorder,
                                      constrained_domain=self.periodicity)
