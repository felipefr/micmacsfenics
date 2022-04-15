#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:04:59 2022

@author: felipe
"""

import sys
import dolfin as df
import numpy as np
sys.path.insert(0, '../../core/')
from micmacsfenics.core.micro_constitutive_model_eps import MicroConstitutiveModelEps
from fenicsUtils import symgrad_voigt
from functools import partial 

from ddfunction import DDFunction



solver_parameters = {"nonlinear_solver": "newton",
                     "newton_solver": {"maximum_iterations": 20,
                                       "report": False,
                                       "error_on_nonconvergence": True}}

resultFolder = './results/'



def getPsi(u, param): # linear elastic one
    C = param[0]    
    e = symgrad_voigt(u)
    return 0.5*df.inner( df.dot(C,e), e)


class myChom(df.UserExpression):
    def __init__(self, microModels,  **kwargs):
        self.microModels = microModels
        self.eps_data = 0.0
        super().__init__(**kwargs)
        
    def setMacrodeformation(self, eps,u):
        self.eps = eps
        self.u = u

    def updateMacrodeformation(self):
        self.eps.update(symgrad_voigt(self.u))
        self.eps_data = self.eps.data()
        
        
    def eval_cell(self, values, x, cell):
        self.updateMacrodeformation()
        values[:] = self.microModels[cell.index].getTangent(self.eps_data[cell.index]).flatten()

    def value_shape(self):
        return (3, 3,)


def getFactorBalls(seed=1):
    Nballs = 4
    ellipseData = np.zeros((Nballs, 3))
    xlin = np.linspace(0.0 + 0.5/np.sqrt(Nballs), 1.0 - 0.5/np.sqrt(Nballs),
                       int(np.sqrt(Nballs)))

    grid = np.meshgrid(xlin, xlin)
    ellipseData[:, 0] = grid[0].flatten()
    ellipseData[:, 1] = grid[1].flatten()
    np.random.seed(seed)
    ellipseData[:, 2] = r0 + np.random.rand(Nballs)*(r1 - r0)

    fac = df.Expression('1.0', degree=0)  # ground substance
    radiusThreshold = 0.01

    str_fac = 'A*exp(-a*((x[0] - x0)*(x[0] - x0) + (x[1] - y0)*(x[1] - y0) ))'

    for xi, yi, ri in ellipseData[:, 0:3]:
        fac = fac + df.Expression(str_fac, A=contrast - 1.0,
                                  a=- np.log(radiusThreshold)/ri**2,
                                  x0=xi, y0=yi, degree=0)

    return fac


Nx = 2
Ny = 2
NxMicro = NyMicro = 5
bndModel = 'lin' 

name_meshmacro = 'meshmacro.xdmf'
name_meshmicro = 'meshmicro.xdmf'

createMesh = False

r0 = 0.3
r1 = 0.5
Lx = 2.0
Ly = 0.5

lamb_matrix = 1.0
mu_matrix = 0.5
LxMicro = LyMicro = 1.0
contrast = 10.0

ty = -0.01



# Create mesh and define function space
if(createMesh):
    mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly),
                            Nx, Ny, "right/left")
    
    with df.XDMFFile(name_meshmacro) as f:
        f.write(mesh)

else:
    mesh = df.Mesh()
    with df.XDMFFile(name_meshmacro) as f:
        f.read(mesh)



Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1)

leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx=Lx)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)

# defining the micro model
nCells = mesh.num_cells()

if(createMesh):
    meshMicro = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(LxMicro, LyMicro),
                                 NxMicro, NyMicro, "right/left")
    
    with df.XDMFFile(name_meshmicro) as f:
        f.write(meshMicro)

else:
    meshMicro = df.Mesh()
    with df.XDMFFile(name_meshmicro) as f:
        f.read(meshMicro)

facs = [getFactorBalls(0) for i in range(nCells)]
np.random.seed(0)
params = [[fac_i*lamb_matrix, fac_i*mu_matrix] for fac_i in facs]
microModels = [MicroConstitutiveModelEps(meshMicro, pi, bndModel)
               for pi in params]

Chom = myChom(microModels, degree=0)

# Define boundary condition
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0)), boundary_markers, 1)

ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
traction = df.Constant((0.0, ty))

# Define variational problem


duh = df.TrialFunction(Uh)            # Incremental displacement
vh  = df.TestFunction(Uh)             # Test function
uh  = df.Function(Uh)                 # Displacement from previous iteration


Sh0 = df.VectorFunctionSpace(mesh, 'DG', degree = 0 , dim = 3) # for stress
eps = DDFunction(Sh0)

Chom.setMacrodeformation(eps, uh)
Chom.updateMacrodeformation()


param = [Chom]
psi_law = partial(getPsi, param = param)

Pi = psi_law(uh)*df.dx - df.inner(traction, uh)*ds(2)
 
F = df.derivative(Pi, uh, vh)
J = df.derivative(F, uh, duh)

problem = df.NonlinearVariationalProblem(F, uh, bcL, J)

solver = df.NonlinearVariationalSolver(problem)
solver.parameters.update(solver_parameters)
solver.solve()




# Save solution in VTK format
solFile =  resultFolder + "bar_multiscale_minimisation.xdmf".format(bndModel)


uh.rename("uh", ".")
with df.XDMFFile(solFile) as f:
    f.write(uh, 0.0)
    
    
uh_ref = df.Function(Uh)
solFile =  resultFolder + "bar_multiscale_standalone_{0}_checkpoint.xdmf".format(bndModel)

with df.XDMFFile(solFile) as f:
    f.read_checkpoint(uh_ref, 'u')

error = np.sqrt(df.assemble( df.inner( uh_ref - uh, uh_ref - uh)*df.dx))
print(" difference: %f"%error)

# # plot microstructures
# fac_h = df.project(facs[0], df.FunctionSpace(meshMicro, 'CG', 1))
# fac_h.rename("constrast", "label")
# df.plot(df.project(facs[0], df.FunctionSpace(meshMicro, 'CG', 1)))
# fileResults_2 = df.XDMFFile("microstructure_2.xdmf")
# fileResults_2.write(fac_h)
