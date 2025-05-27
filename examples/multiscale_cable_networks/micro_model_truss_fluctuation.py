#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:08:33 2025

@author: frocha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 19:49:17 2025

@author: frocha
"""

# To Do: - In terms of fluctuations (better convergence?)
# - store u for each gauss point : done (it does not work)
# - initial guess truss : done (it does not work)
# - overrelaxed Newton  : done (it does not work) 

import sys
sys.path.append('/home/frocha/sources/pyola/')
sys.path.append("/home/frocha/sources/fetricksx/")
import fetricksx as ft
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import copy
from toy_solver import * 
from micro_model_truss import MicroModelTruss,  TrussLocalCanonicalIntegrator

class MicroModelTrussFluctuation(MicroModelTruss):        
    def get_ulift(self, G):
        G2x2 = ft.unsym2tensor_np(G)
        ulift = np.zeros_like(self.mesh.X)
        for i in range(self.mesh.X.shape[0]):
            ulift[i,:] = G2x2@(self.mesh.X[i,:] - self.yG)
    
        return ulift.flatten()
    
    def solve_microproblem(self, G, u0 = None, update_u = True):
        self.ulift = Function(self.U)        
        self.utot = Function(self.U)
        self.ulift.array = self.get_ulift(G)
        u0 = copy.deepcopy(self.u) 
        forces = np.zeros_like(u0)
        u =  solve_nonlinear_lift(self.mesh, self.U, self.dh, self.form, forces, 
                             self.bcs, uold = u0, ulift = self.ulift, tol = 1e-8, omega = 0.8, log = False)
        if(update_u):
            self.u.array = u.array
            self.G_last[:] = G[:]
            self.utot.array = self.ulift.array + self.u.array
        return u

    def homogenise_stress(self):        
        return self.homogeniseP_given_disp(self.utot)    

    # Analytical tangent in "unsym flattening": (00,11,01,10) order
    def homogenise_tangent(self):
        return self.homogenise_tangent_given_disp(self.utot) 
        
    def homogenise_stress_solve(self, G, u0 = None): 
        u = self.solve_microproblem(G, u0, update_u = False)
        utot = Function(self.U)
        utot.array = self.ulift.array + u.array
        P = self.homogeniseP_given_disp(utot)    
        return P


   
