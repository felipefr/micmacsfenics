#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 12:48:38 2025

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2025, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or <felipe.f.rocha@gmail.com>
"""

import dolfin as df
import ufl
import fetricks as ft
from micmacsfenics.formulations.multiscale_formulation import MultiscaleFormulation


class FormulationMinimallyConstrainedHighOrder(MultiscaleFormulation):

    def otherSpaces(self):
        return [df.TensorFunctionSpace(self.mesh, "Real", 0),
                df.TensorFunctionSpace(self.mesh, "Real", 0, (2,2,2))]

    def otherRestrictions(self):
        return [None, None]

    def blocks(self):
        aa, ff = super(FormulationMinimallyConstrainedHighOrder, self).blocks()
        
        n = df.FacetNormal(self.mesh)
        i,j,k,l = ufl.indices(4)
        
        y = df.SpatialCoordinate(self.mesh)
        Jinv = self.others['Jinv']
        
        if 'external_bnd' in self.others:
             def weak_constraintG(Lamb,w):
                 aux = ufl.as_tensor(w[i]*n[j], (i,j))
                 return sum([df.inner(Lamb, aux)*self.mesh.ds(kk)
                             for kk in self.others['external_bnd']])
             def weak_constraintH(Lamb,w):
                 aux = ufl.as_tensor(w[i]*n[j]*y[l]*Jinv[l,k], (i,j,k))
                 return sum([df.inner(Lamb, aux)*self.mesh.ds(kk)
                             for kk in self.others['external_bnd']])
        else:
             def weak_constraintG(Lamb,w):
                 return df.inner(Lamb, ufl.as_tensor(w[i]*n[j], (i,j)))*self.mesh.ds
             def weak_constraintH(Lamb,w):
                 aux = ufl.as_tensor(w[i]*n[j]*y[l]*J[l,k], (i,j,k))
                 return df.inner(Lamb, aux)*self.mesh.ds

        u, P, PHO = self.uu_[0], self.uu_[2], self.uu_[3]
        v, Q, QHO = self.vv_[0], self.vv_[2], self.vv_[3]
        
        aa[0].append(- weak_constraintG(P,v))
        aa[0].append(- weak_constraintH(PHO,v))
        aa[1].append(0)
        aa[1].append(0)
        aa.append([- weak_constraintG(Q,u), 0, 0, 0])
        aa.append([- weak_constraintH(QHO,u), 0, 0, 0])

        ff.append(0)
        ff.append(0)

        return [aa, ff]
