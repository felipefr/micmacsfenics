#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Availabel in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com, felipe.figueredorocha@epfl.ch

Bar problem given a Multiscale constitutive law:
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and traction on the right.
The constitutive law is given implicitly by solving a micro problem in each gauss point of micro-scale. (one per element todo: one per GP)
We can choose the kinematically constrained model to the micro problem:
Linear, Periodic or Minimally Restricted
Entry to constitutive law: Mesh (micro), Lamé parameters (variable in micro), Kinematical Model    
"""

import sys, os
from dolfin import *
import matplotlib.pyplot as plt
from ufl import nabla_div
sys.path.insert(0, '../utils/')
import multiscaleModels as mscm
from fenicsUtils import symgrad, symgrad_voigt
import numpy as np

class myChom(UserExpression):
    def __init__(self, microModels,  **kwargs):
        self.microModels = microModels
        
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, cell):    
        values[:] = self.microModels[cell.index].getTangent().flatten()
        
    def value_shape(self):
        return (3,3,)
    

def getFactorBalls(seed = 1):
    Nballs = 4
    ellipseData = np.zeros((Nballs,3))
    xlin = np.linspace(0.0 + 0.5/np.sqrt(Nballs) ,1.0 - 0.5/np.sqrt(Nballs), int(np.sqrt(Nballs)))
    grid = np.meshgrid(xlin,xlin)
    ellipseData[:,0] = grid[0].flatten() 
    ellipseData[:,1] = grid[1].flatten() 
    np.random.seed(seed)
    ellipseData[:,2] = r0 + np.random.rand(Nballs)*(r1 - r0)
    
    fac = Expression('1.0', degree = 2) # ground substance=
    radiusThreshold = 0.01
    
    for xi, yi, ri in ellipseData[:,0:3]:
        fac = fac + Expression('A*exp(-a*( (x[0] - x0)*(x[0] - x0) + (x[1] - y0)*(x[1] - y0) ) )', 
                               A = contrast - 1.0, a = - np.log(radiusThreshold)/ri**2, 
                               x0 = xi, y0 = yi, degree = 2)
        
    return fac

r0 = 0.3
r1 = 0.5
Lx = 2.0
Ly = 0.5
Nx = 2
Ny = 2

lamb_matrix = 1.0
mu_matrix = 0.5
NxMicro = NyMicro = 10
LxMicro = LyMicro = 1.0
contrast = 10.0

ty = -0.01



# Create mesh and define function space
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny, "right/left")
Uh = VectorFunctionSpace(mesh, "Lagrange", 1)

leftBnd = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx = Lx)

boundary_markers = MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)

# defining the micro model
nCells = mesh.num_cells()
facs = [getFactorBalls(i) for i in range(nCells)]
meshMicro = RectangleMesh(Point(0.0, 0.0), Point(LxMicro, LyMicro), NxMicro, NyMicro, "right/left")
Chom = myChom([mscm.MicroConstitutiveModel(meshMicro, [facs[i]*lamb_matrix,facs[i]*mu_matrix], 'per') for i in range(nCells)], degree = 0)

# Define boundary condition
bcL = DirichletBC(Uh, Constant((0.0,0.0)), boundary_markers, 1) # leftBnd instead is possible

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
traction = Constant((0.0,ty ))


# Define variational problem
uh = TrialFunction(Uh) 
vh = TestFunction(Uh)
a = inner(dot(Chom,symgrad_voigt(uh)), symgrad_voigt(vh))*dx
b = inner(traction,vh)*ds(2)

# Compute solution
uh = Function(Uh)
solve(a == b, uh, bcL)

# Save solution in VTK format
fileResults = XDMFFile("barMultiscale_2micro_single.xdmf")
fileResults.write(uh)



fac_h = project(facs[0], FunctionSpace(meshMicro, 'CG', 1))
fac_h.rename("constrast", "label")
plot(project(facs[0], FunctionSpace(meshMicro, 'CG', 1)))
fileResults_2 = XDMFFile("microstructure_2.xdmf")
fileResults_2.write(fac_h)


# print(uh(Point(2.0,0.0)))


