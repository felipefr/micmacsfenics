#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:57:46 2021

@author: felipefr
"""

import dolfin as df
from multiscale_formulation import MultiscaleFormulation


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
