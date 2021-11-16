#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 21:00:32 2021

@author: felipefr
"""

import dolfin as df
from micmacsfenics.formulations.multiscale_formulation import MultiscaleFormulation


class FormulationMinimallyConstrained(MultiscaleFormulation):

    def otherSpaces(self):
        return [df.TensorFunctionSpace(self.mesh, "Real", 0)]

    def otherRestrictions(self):
        return [None]

    def blocks(self):
        aa, ff = super(FormulationMinimallyConstrained, self).blocks()

        n = df.FacetNormal(self.mesh)
        ds = df.Measure('ds', self.mesh)

        u, P = self.uu_[0], self.uu_[2]
        v, Q = self.vv_[0], self.vv_[2]

        aa[0].append(- df.inner(P, df.outer(v, n))*ds)
        aa[1].append(0)
        aa.append([- df.inner(Q, df.outer(u, n))*ds, 0, 0])

        ff.append(0)

        return [aa, ff]
