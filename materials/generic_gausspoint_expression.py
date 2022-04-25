#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:49:07 2022

@author: felipe
"""

import dolfin as df

class genericGaussPointExpression(df.UserExpression):
    def __init__(self, strain, homogenisationLaw, shape,  **kwargs):
        self.strain = strain
        self.homogenisationLaw = homogenisationLaw
        self.shape = shape
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        strain = self.strain.vector().get_local()[cell.index*3:(cell.index + 1)*3]
        values[:] = self.homogenisationLaw(strain, cell).flatten()

    def value_shape(self):
        return self.shape