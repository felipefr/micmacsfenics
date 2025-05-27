#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 19:49:17 2025

@author: frocha
"""
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

class TrussLocalCanonicalIntegrator:
    def __init__(self, A, material, V):
        self.A = A
        self.material = material
        self.element = V.element
        self.V = V
        self.k = 0
        self.l = 0
    
    def set_kl(self, k, l):
        self.k, self.l = k, l
        
    def compute(self, X, u, uold, e):
        A = self.A[e]
        a, b, lmbda, lmbda_old, L0, Bmat = self.element.get_truss_kinematics(X, u, uold)
        
        stress, dtang = self.material(lmbda, lmbda_old)
        
        V = L0*A  # Volume

        # Internal force vector (2D)
        D = dtang * np.outer(b,b) 
        D += stress*(np.eye(2) - np.outer(b,b))/lmbda
        

        # Tangent stiffness matrix (4x4)
        K = V * Bmat.T @ D @ Bmat
        f_int = - V * a[self.l] * Bmat.T @ D[:, self.k] 
        
        return K, f_int


class MicroModelTruss:
    def __init__(self, mesh, param):
        self.mesh = mesh

        if(param['model'] == 'truss'):
            self.material = LinearEngStrainTrussMaterial(E=param['E'])
            self.material_can = LinearEngStrainTrussMaterial(E=param['E'])
            
        elif(param['model'] == 'cable'):
            self.material = LinearEngStrainCableMaterial(E=param['E'], eta = param['eta'])
            self.material_can = LinearEngStrainCableMaterial(E=param['E'], eta = 0.0)
        
        self.U = FunctionSpace(self.mesh, TrussElement())
        self.dh = DOFHandler(self.mesh)
        self.dh.add_space(self.U, name = 'displacement')
        self.form = TrussLocalIntegrator(param['A'], self.material, self.U)
        self.form_can = TrussLocalCanonicalIntegrator(param['A'], self.material_can, self.U)
        
        self.ngamma_nodes = len(mesh.bnd_nodes)
        uD = np.zeros((self.ngamma_nodes, 2))
        self.bcs = [DirichletBC(mesh.bnd_nodes, [0, 1], uD)]
        
        self.yG = np.array([0.5,0.5])
        self.vol = 1.0
        self.u = Function(self.U)
        self.G_last = np.zeros(4)
        
    def get_ufixed(self, G):
        G2x2 = ft.unsym2tensor_np(G)
        uD = np.zeros_like(self.bcs[0].value)
        for i, j in enumerate(self.mesh.bnd_nodes):
            uD[i,:] = G2x2@(self.mesh.X[j,:] - self.yG)
    
        return uD
    
    def solve_microproblem(self, G, u0 = None, update_u = True):        
        self.bcs[0].value = self.get_ufixed(G)
        u0 = Function(self.U) if type(u0) == type(None) else copy.deepcopy(u0)
        forces = np.zeros_like(u0)
        u =  solve_nonlinear(self.mesh, self.U, self.dh, self.form, forces, 
                             self.bcs, uold = u0, tol = 1e-8, log = False)
        if(update_u):
            self.u.array = u.array
            self.G_last[:] = G[:]
        return u

    # this is done once the microproblem is solved
    def get_stress_tangent(self):
        return ft.tensor2unsym(self.homogenise_stress()), ft.sym_flatten_4x4_np(self.homogenise_tangent())
        # return ft.tensor2unsym(self.homogenise_stress()), ft.sym_flatten_4x4_np(self.homogenise_tang_ffd())

    def homogenise_stress(self):        
        P = self.homogeniseP_given_disp(self.u)    
        return P

    def homogenise_stress_solve(self, G, u0 = None): 
        u = self.solve_microproblem(G, u0, update_u = False)
        P = self.homogeniseP_given_disp(u)    
        return P
    
    
    def homogeniseP_given_disp(self, u):
        P = np.zeros((2,2))
        for c in range(self.mesh.n_cells):
            X = self.mesh.X[self.mesh.cells[c]]
            cell_dofs = self.dh.get_cell_dofs(c)[0] # only for the first space (U)
            uL = u.array[cell_dofs]
        
            A = self.form.A[c]
            a, b, lmbda, lmbda_old, L0, Bmat = self.form.element.get_truss_kinematics(X.flatten(), uL, uL) # last argument is dummy
            stress, dtang = self.material(lmbda, lmbda_old)
            
            V = L0*A  # Volume            
            P += V*stress*np.outer(b,a)
        
        P = P/self.vol
        
        return P

    # Analytical tangent in "unsym flattening": (00,11,01,10) order
    def homogenise_tangent(self):
        # easier to work in the lexigraphic order and then permute
        ass = Assembler(self.mesh, self.dh)
        
        self.bcs[0].value[:,:] = 0.0
        ukl_list = []
        for k in range(2):
            for l in range(2):
                self.form_can.set_kl(k, l)
                K, F_kl = ass.assemble(self.form_can, self.u, self.u)
                for bc in self.bcs:
                    bc.apply(K,F_kl)
                
                K.tocsr()
                ukl_list.append(spla.spsolve(K, F_kl))
    
        C = np.zeros((4,4))
        for c in range(self.mesh.n_cells):
            X = self.mesh.X[self.mesh.cells[c]]
            cell_dofs = self.dh.get_cell_dofs(c)[0] # only for the first space (U)
            uL = self.u.array[cell_dofs]
            
            A = self.form.A[c]
            a, b, lmbda, lmbda_old, L0, Bmat = self.form.element.get_truss_kinematics(X.flatten(), uL, uL) # last argument is dummy
            stress, dtang = self.material(lmbda, lmbda_old)
            
            D = dtang * np.outer(b,b) 
            D += stress*(np.eye(2) - np.outer(b,b))/lmbda
            
            V = L0*A  # Volume        
            
            # Cbar (note that reshape respect the lexigraphic order)
            C += V*np.einsum("ik,j,l->ijkl", D, a, a).reshape((4,4)) 
            
            for kl in range(4):
                uklL = ukl_list[kl][cell_dofs]
                C[:,kl] += V* np.outer(D@Bmat@uklL,a).flatten() 
            
        # permutation for the "unsym order"
        perm = np.array([0,3,1,2]).astype('int')
        C = C[perm,:][:,perm]
            
        return C

    # Numerical tangent by forward finite differences in "unsym flattening": (00,11,01,10) order
    def homogenise_tang_ffd(self, G = None, tau = 1e-7):
        P_ref = self.homogeniseP_given_disp(self.u)
        P_ref = ft.tensor2unsym_np(P_ref)
        Gref = self.G_last[:] if type(G) == type(None) else copy.deepcopy(G) 
        n = len(Gref) 
        base_canonic = np.eye(n)
        Atang = np.zeros((n,n))
        
        for j in range(n):
            Gp = Gref + tau*base_canonic[j,:]
            Pp  = ft.tensor2unsym_np(self.homogenise_stress_solve(Gp, u0 = self.u))
            Atang[:,j] = (Pp - P_ref)/tau 
        
        return Atang
        
   
