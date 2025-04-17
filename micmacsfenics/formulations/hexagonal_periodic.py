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
        self.d = 2*a/np.sqrt(3.) # distance between two opposite nodes (diameter)
        self.angles = np.linspace(phase, phase + 2*np.pi/3., 3)
        
        # unit vector periodicities (faces)
        self.v = np.array([[np.cos(ang), np.sin(ang)] for ang in self.angles])
        self.w = np.array( [ (self.v[0]+self.v[1])/np.linalg.norm(self.v[0]+self.v[1]),
                             (self.v[1]+self.v[2])/np.linalg.norm(self.v[1]+self.v[2]),
                             (self.v[2]-self.v[0])/np.linalg.norm(self.v[2]-self.v[0])])
        
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        # return True if on face 0,1 or 2, excluding corners 0-5 and 2-3
        if(on_boundary):
            one_of_master_faces = self.is_on_face_i(x, 0) or (self.is_on_face_i(x, 1) or self.is_on_face_i(x, 2))
            node3 = self.is_on_face_i(x, 2) and self.is_on_face_i(x, 3)
            return (one_of_master_faces and not node3)

        return False

    def is_on_face_i(self, x, i):
        ang = self.angles[i] if i<3 else (self.angles[i-3] + np.pi)
        # (R@x)_1 (the only important component). R is clock-wise rotation
        d = np.dot(np.array([np.cos(ang),np.sin(ang)]) , x)
        return df.near(d, 0.5*self.a)
    
#    wrong corner mapping
    def map(self, x, y):
        if((self.is_on_face_i(x, 3) and self.is_on_face_i(x, 4)) or (self.is_on_face_i(x, 4) and self.is_on_face_i(x, 5)) ):
            y[:] = x[:] + self.a*self.v[1,:] 
        elif(self.is_on_face_i(x, 0) and self.is_on_face_i(x, 5)):
            y[:] = x[:] + self.a*self.v[2,:]
        elif(self.is_on_face_i(x, 2) and self.is_on_face_i(x, 3)):
            y[:] = x[:] + self.a*self.v[0,:]
        elif(self.is_on_face_i(x, 3)):
            y[:] = x[:] + self.a*self.v[0,:]
        elif(self.is_on_face_i(x, 4)):
            y[:] = x[:] + self.a*self.v[1,:]
        elif(self.is_on_face_i(x, 5)):
            y[:] = x[:] + self.a*self.v[2,:]
        else:
            y[0] = x[0]
            y[1] = x[1]

    # def map(self, x, y):
    #     node3 = self.is_on_face_i(x, 2) and self.is_on_face_i(x, 3)
    #     node4 = self.is_on_face_i(x, 3) and self.is_on_face_i(x, 4)
    #     node5 = self.is_on_face_i(x, 4) and self.is_on_face_i(x, 5)
        
    #     if(node3):
    #         y[:] = x[:] - self.d*self.w[2,:]
    #     elif(node4):
    #         y[:] = x[:] + self.d*self.w[0,:]
    #     elif(node5):
    #         y[:] = x[:] + self.d*self.w[1,:]                
    #     elif(self.is_on_face_i(x, 3)):
    #         y[:] = x[:] + self.a*self.v[0,:]
    #     elif(self.is_on_face_i(x, 4)):
    #         y[:] = x[:] + self.a*self.v[1,:]
    #     elif(self.is_on_face_i(x, 5)):
    #         y[:] = x[:] + self.a*self.v[2,:]
    #     else:
    #         y[0] = x[0]
    #         y[1] = x[1]


# similar values but higher values where should be zero
    # def map(self, x, y):
    #     y[0] = x[0]
    #     y[1] = x[1]
    #     for i in range(3,6):
    #         if(self.is_on_face_i(x, i)):
    #             y[:] += self.a*self.v[i-3,:]
        

class FormulationHexagonalPeriodic(MultiscaleFormulation):

    def flutuationSpace(self):
        polyorder = self.others['polyorder']
        self.periodicity = HexagonalPeriodicBoundary()

        return df.VectorFunctionSpace(self.mesh, "CG", polyorder,
                                      constrained_domain=self.periodicity)
