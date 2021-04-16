#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:47:54 2021

@author: felipefr
"""

import numpy as np
import multiphenics as mp
import dolfin as df
from timeit import default_timer as timer
from ufl import nabla_div
from fenicsUtils import symgrad, Integral, symgrad_voigt

def macro_strain(i):
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return np.array([[Eps_Voigt[0], Eps_Voigt[2]/2.], 
                    [Eps_Voigt[2]/2., Eps_Voigt[1]]])
def stress2Voigt(s):
    return df.as_vector([s[0,0], s[1,1], s[0,1]])

def strain2Voigt(e):
    return df.as_vector([e[0,0], e[1,1], 2*e[0,1]])

class PeriodicBoundary(df.SubDomain):
    # Left boundary is "target domain" G
    def __init__(self,x0 = 0.0,x1 = 1.0,y0 = 0.0 ,y1 = 1.0, **kwargs):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        
        super().__init__(**kwargs)
    
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        if(on_boundary):
            left, bottom, right, top = self.checkPosition(x)
            return (left and not top) or (bottom and not right)
        
        return False
     
    def checkPosition(self,x):
        return df.near(x[0], self.x0), df.near(x[1],self.y0), df.near(x[0], self.x1), df.near(x[1], self.y1)
    
    def map(self, x, y):
        left, bottom, right, top = self.checkPosition(x)
        
        y[0] = x[0] + self.x0 - (self.x1 if right else self.x0)
        y[1] = x[1] + self.y0 - (self.y1 if top else self.y0)


class MultiscaleFormulation:
    
    def __init__(self, mesh, sigma, Eps, others):
        
        self.mesh = mesh
        self.others = others
        self.sigma = sigma
        self.Eps = Eps
        
        V = self.flutuationSpace() 
        R = self.zeroAverageSpace()
        self.W = mp.BlockFunctionSpace([V,R] + self.otherSpaces())
        
        self.uu = mp.BlockTrialFunction(self.W)
        self.vv = mp.BlockTestFunction(self.W)
        self.uu_ = mp.block_split(self.uu)
        self.vv_ = mp.block_split(self.vv)
                
    def __call__(self):
        return self.blocks() + self.bcs() + [self.W] 

    def blocks(self):
        dx = df.Measure('dx', self.mesh)
        x = df.SpatialCoordinate(self.mesh)
       
        u, p = self.uu_[0:2] 
        v, q = self.vv_[0:2]
        
        aa = [[df.inner(self.sigma(u), symgrad(v))*dx , df.inner(p,v)*dx], [df.inner(q,u)*dx , 0]]
        ff = [-df.inner(self.Eps, self.sigma(v))*dx, 0]    # dot(sigma(Eps) , symgrad(v)) = dot(Eps , sigma(symgrad(v))) 
        
        return [aa, ff]
    
    def bcs(self):
        return [[]]
    
    def enrichBlocks(self):
        pass

    def createMixedSpace(self):
        pass        

    def flutuationSpace(self):
        return df.VectorFunctionSpace(self.mesh,"CG", self.others['polyorder'])
    
    def zeroAverageSpace(self):
        return df.VectorFunctionSpace(self.mesh, "Real", 0)

    def otherSpaces(self):
        return []


class MultiscaleFormulationMR(MultiscaleFormulation):

    def otherSpaces(self):
        return [df.TensorFunctionSpace(self.mesh, "Real", 0)]
        
    def blocks(self):
        aa , ff = super(MultiscaleFormulationMR, self).blocks()
    
        n = df.FacetNormal(self.mesh)
        ds = df.Measure('ds', self.mesh)
        
        u, P = self.uu_[0], self.uu_[2]
        v, Q = self.vv_[0], self.vv_[2]
    
        aa[0].append(- df.inner(P,df.outer(v,n))*ds)
        aa[1].append(0)
        aa.append([- df.inner(Q,df.outer(u,n))*ds, 0, 0])

        ff.append(0) 
        
        return [aa, ff]
        
class MultiscaleFormulationPeriodic(MultiscaleFormulation):
    def flutuationSpace(self):
        polyorder = self.others['polyorder']
        periodicity = PeriodicBoundary(self.others['x0'] ,self.others['x1'],self.others['y0'],self.others['y1'])
        return df.VectorFunctionSpace(self.mesh,"CG", polyorder, constrained_domain = periodicity )
        
class MultiscaleFormulationLin(MultiscaleFormulation):
    def bcs(self):
        onBoundary = df.CompiledSubDomain('on_boundary')
        uD = self.others['uD'] if 'uD' in self.others else df.Constant((0.,0.)) 
        bc1 = mp.DirichletBC(self.W.sub(0), uD , onBoundary)
        return [mp.BlockDirichletBC([bc1])]
        
listMultiscaleModels = {'MR' : MultiscaleFormulationMR, 
                        'per': MultiscaleFormulationPeriodic,
                        'lin': MultiscaleFormulationLin}

class MicroConstitutiveModel:
    
    def __init__(self,mesh,lame, model):
        self.sigmaLaw = lambda u: lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*symgrad(u)
        self.mesh = mesh
        self.model = model
        self.coord_min = np.min(self.mesh.coordinates(), axis = 0)
        self.coord_max = np.max(self.mesh.coordinates(), axis = 0)
        
        # it should be modified before computing tangent (if needed) 
        self.others = {'polyorder' : 1, 'x0': self.coord_min[0], 'x1': self.coord_max[0], 
                                        'y0': self.coord_min[1], 'y1': self.coord_max[1]} 
        
        self.multiscaleModel = listMultiscaleModels[model]
        self.x = df.SpatialCoordinate(self.mesh)               
        self.ndim = 2
        self.nvoigt = int(self.ndim*(self.ndim + 1)/2)  
        self.Chom_ = np.zeros((self.nvoigt,self.nvoigt))

        self.getTangent = self.computeTangent # in the first run should compute 
        
    def computeTangent(self):
        
        dy = df.Measure('dx',self.mesh)
        vol = df.assemble(df.Constant(1.0)*dy)
        y = df.SpatialCoordinate(self.mesh)
        Eps = df.Constant(((0.,0.),(0.,0.))) # just placeholder
        
        form = self.multiscaleModel(self.mesh, self.sigmaLaw, Eps, self.others)
        a,f,bcs,W = form()

        start = timer()        
        A = mp.block_assemble(a)
        if(len(bcs) > 0): 
            bcs.apply(A)
        
        solver = df.PETScLUSolver('superlu') # decompose just once (the faster for single process)
        sol = mp.BlockFunction(W)
        
        end = timer()
        print('time assembling system', end - start) # Time in seconds
        
        for i in range(self.nvoigt):
            start = timer()     
            Eps.assign(df.Constant(macro_strain(i)))    
            F = mp.block_assemble(f)
            if(len(bcs) > 0):
                bcs.apply(F)
        
            solver.solve(A,sol.block_vector(), F)    
            sol.block_vector().block_function().apply("to subfunctions")
            
            sig_mu = self.sigmaLaw(df.dot(Eps,y) + sol[0])
            sigma_hom =  Integral(sig_mu, dy, (2,2))/vol

            self.Chom_[:,i] = sigma_hom.flatten()[[0,3,1]]
            
            end = timer()
            print('time in solving system', end - start) # Time in seconds
        
        print(self.Chom_)
        
        self.getTangent = self.getTangent_ # from the second run onwards, just returns  
        
        return self.Chom_
               
    def getTangent_(self):
        return self.Chom_
    
    def solveStress(self,u):
        return df.dot( df.Constant(self.getTangent()) , symgrad_voigt(u))
