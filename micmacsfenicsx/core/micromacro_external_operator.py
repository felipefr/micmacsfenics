#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 00:38:37 2024

@author: felipe


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import numpy as np
import ufl

import fetricksx as ft
from dolfinx_external_operator import (FEMExternalOperator, replace_external_operators, 
                                       evaluate_operands, evaluate_external_operators)
    
# most straightforward implementation
class MicroMacroExternalOperator:
    
    def __init__(self, operand, W, Wtan, micromodels=None):        
        self.stress = FEMExternalOperator(operand, function_space=W.space, 
                      external_function = lambda d: self.stress_impl if d == (0,) else NotImplementedError)
        
        # no need to implement external_function because stress_impl already 
        # compute the tangent for the sake of perfomance
        self.tangent = FEMExternalOperator(operand, function_space=Wtan.space)

        self.tangent_array = self.tangent.ref_coefficient.x.array.reshape((Wtan.nq_mesh, Wtan.space.num_sub_spaces))
        self.stress_array = self.stress.ref_coefficient.x.array.reshape((W.nq_mesh, W.space.num_sub_spaces))
        
        self.micromodels = micromodels
        
    def stress_impl(self, strain):
        for s, t, e, m in zip(self.stress_array, self.tangent_array, strain, self.micromodels):
            m.solve_microproblem(e) 
            s[:], t[:] = m.get_stress_tangent()  
                 
        return self.stress_array.reshape(-1)


    def tangent_op(self, de):
        return ufl.dot(ft.as_sym_tensor_3x3(self.tangent), de) 
    
    def register_forms(self, res_ext, J_ext):        
        res_replaced, self.F_external_operators = replace_external_operators(res_ext)
        # J_external_operators are not used because it is already included on stress evaluation
        J_replaced, _ = replace_external_operators(J_ext) 
        return res_replaced, J_replaced
    
    def update(self, dummy1, dummy2):
        evaluated_operands = evaluate_operands(self.F_external_operators)
        # Probably unecessary allocation here, since stress and tangent vectors are already allocated elsewhere
        _ = evaluate_external_operators(self.F_external_operators, evaluated_operands)
        # no need of evaluation of J_external
        # _ = evaluate_external_operators(self.J_external_operators, evaluated_operands)


# Implementation using just one object
# class MicroMacroExternalOperator:
    
#     def __init__(self, operand, Wstress_tan, micromodels=None):        
#         self.stress_tangent = FEMExternalOperator(operand, function_space=Wstress_tan.space, 
#                       external_function = lambda d: self.stress_tan_impl if d == (0,) else NotImplementedError)

#         self.micromodels = micromodels
        
#         self.stress_tangent_array = self.stress_tangent.ref_coefficient.x.array.reshape((Wstress_tan.nq_mesh, Wstress_tan.space.num_sub_spaces ))
#         self.stress = ufl.as_vector([self.stress_tangent[i] for i in range(3)])
#         self.tangent = ufl.as_vector([self.stress_tangent[i] for i in range(3,9)]) 

#     def stress_tan_impl(self, strain):
#         return np.array([m.get_stress_tangent_solve(e) for e, m in zip(strain, self.micromodels)]).reshape(-1)

#     def tangent_op(self, de):
#         return ufl.dot(ft.as_sym_tensor_3x3(self.tangent), de) 
    
#     def register_forms(self, res_ext, J_ext):        
#         res_replaced, self.F_external_operators = replace_external_operators(res_ext)
#         J_replaced, self.J_external_operators = replace_external_operators(J_ext)
#         return res_replaced, J_replaced,
    
#     def update(self, dummy1, dummy2):
#         evaluated_operands = evaluate_operands(self.F_external_operators)
#         _ = evaluate_external_operators(self.F_external_operators, evaluated_operands)


