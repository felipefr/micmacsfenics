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
from micmacsfenics.formulations.multiscale_formulation import MultiscaleFormulation


class PeriodicBoundary(df.SubDomain):
    # Left boundary is "target domain" G
    def __init__(self, x0=0.0, x1=1.0, y0=0.0, y1=1.0, **kwargs):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT
        # on one of the two corners (0, 1) and (1, 0)
        if(on_boundary):
            left, bottom, right, top = self.checkPosition(x)
            return (left and not top) or (bottom and not right)

        return False

    def checkPosition(self, x):
        
        return [df.near(x[0], self.x0), df.near(x[1], self.y0),
                df.near(x[0], self.x1), df.near(x[1], self.y1)]

    def map(self, x, y):
        left, bottom, right, top = self.checkPosition(x)

        y[0] = x[0] + self.x0 - (self.x1 if right else self.x0)
        y[1] = x[1] + self.y0 - (self.y1 if top else self.y0)


class FormulationPeriodic(MultiscaleFormulation):

    def flutuationSpace(self):
        polyorder = self.others['polyorder']
        periodicity = PeriodicBoundary(self.others['x0'], self.others['x1'],
                                       self.others['y0'], self.others['y1'])

        return df.VectorFunctionSpace(self.mesh, "CG", polyorder,
                                      constrained_domain=periodicity)


# Alternative implementation
# def __init__
#     # vector mapping the right surface to the left
#     self.vR2L = (self.x1 - self.x0)*np.array([-1., 0.])  
#     # vector mapping the top surface to the bottom
#     self.vT2B = (self.y1 - self.y0)*np.array([0., -1.])  
    
#     super().__init__(**kwargs)

# def inside(self, x, on_boundary):
#     # return True if on left or bottom boundary AND NOT
#     # on one of the two corners Bottom-right or Left-top
#     if(on_boundary):
#         return ( (self.is_left(x) and not self.is_top(x)) or 
#                  (self.is_bottom(x) and not self.is_right(x)) )

#     return False

# def is_left(self, x):
#     return df.near(x[0], self.x0)

# def is_right(self, x):
#     return df.near(x[0], self.x1)

# def is_bottom(self, x):
#     return df.near(x[1], self.y0)

# def is_top(self, x):
#     return df.near(x[1], self.y1)

# def map(self, x, y):
#     if(self.is_right(x) and self.is_top(x)):
#         y[:] = x[:] + self.vR2L + self.vT2B 
#     elif(self.is_right(x)):
#         y[:] = x[:] + self.vR2L
#     elif(self.is_top(x)):
#         y[:] = x[:] + self.vT2B
#     else:
#         y[0] = x[0]
#         y[1] = x[1]
        