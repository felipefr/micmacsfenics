#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 22:37:17 2024

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 13:11:17 2022

@author: felipe rocha

This file is part of micmacsfenics, a FEniCs-based implementation of 
two-level finite element simulations (FE2) using computational homogenization.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

import numpy as np
import micmacsfenics as mm
import fetricks as ft

class MicroConstitutiveModelMonoscale(mm.MicroConstitutiveModelAbstract): 

    def __init__(self, param):
        self.param = param
        self.lame = [param["mu_ref"], param["lamb_ref"]]
        self.alpha = param["alpha"]
        
        super().__init__(param)

    def __homogeniseTangent(self):
        
        
        tr_e = ft.tr_mandel(self.eps)
        e2 = np.dot(self.eps, self.eps)
        
        self.tangenthom = self.lame[0]*(1 + 3*self.alpha*tr_e**2)*tr_e*np.outer(ft.Id_mandel_np, ft.Id_mandel_np)
        self.tangenthom += 2*self.lame[1]*(1* + self.alpha*e2)*np.eye(self.nvoigt) + 4*self.lame[1]*self.alpha*np.outer(self.eps, self.eps)
        
        
        type(self).countComputeCanonicalProblem = type(self).countComputeCanonicalProblem + 1
            
    def __homogeniseStress(self):
        
        tr_e = ft.tr_mandel(self.eps)
        e2 = np.dot(self.eps, self.eps)
    
        self.stresshom = self.lame[0]*(1 + self.alpha*tr_e**2)*tr_e*ft.Id_mandel_np + 2*self.lame[1]*(1 + self.alpha*e2)*self.eps   
