#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:54:31 2021

@author: felipefr
"""

import multiphenics as mp
import dolfin as df
from multiscale_formulation import MultiscaleFormulation


class FormulationDirichletLagrange(MultiscaleFormulation):
    def otherSpaces(self):
        return [self.flutuationSpace()]

    def otherRestrictions(self):
        onBoundary = df.CompiledSubDomain('on_boundary')
        return [mp.MeshRestriction(self.mesh, onBoundary)]

    def blocks(self):
        aa, ff = super(FormulationDirichletLagrange, self).blocks()

        uD = self.others['uD'] if 'uD' in self.others else df.Constant((0, 0))

        ds = df.Measure('ds', self.mesh)

        u, p = self.uu_[0], self.uu_[2]
        v, q = self.vv_[0], self.vv_[2]

        aa[0].append(df.inner(p, v)*ds)
        aa[1].append(0)
        aa.append([df.inner(q, u)*ds, 0, 0])

        ff.append(df.inner(q, uD)*ds)

        return [aa, ff]
