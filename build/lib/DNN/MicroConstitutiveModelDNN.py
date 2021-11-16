import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div
sys.path.insert(0, '/home/felipefr/github/micmacsFenics/utils/')
sys.path.insert(0,'../utils/')

import multiscaleModels as mscm
from fenicsUtils import symgrad, symgrad_voigt, Integral
import numpy as np

import fenicsMultiscale as fmts
import myHDF5 as myhd
import meshUtils as meut
import elasticity_utils as elut
import symmetryLib as symlpy
from timeit import default_timer as timer
import multiphenics as mp
import fenicsUtils as feut

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()


class MicroConstitutiveModelDNN(mscm.MicroConstitutiveModel):
    
    def __init__(self,nameMesh, param, model):
        self.nameMesh = nameMesh
        self.param = param
        self.model = model
        # it should be modified before computing tangent (if needed) 
        self.others = {'polyorder' : 2} 
        
        self.multiscaleModel = mscm.listMultiscaleModels[model]              
        self.ndim = 2
        self.nvoigt = int(self.ndim*(self.ndim + 1)/2)  
        self.Chom_ = np.zeros((self.nvoigt,self.nvoigt))
 
        self.getTangent = self.computeTangent # in the first run should compute     
            
    def readMesh(self):
        self.mesh = meut.EnrichedMesh(self.nameMesh,comm_self)
        self.lame = elut.getLameInclusions(*self.param, self.mesh)
        self.coord_min = np.min(self.mesh.coordinates(), axis = 0)
        self.coord_max = np.max(self.mesh.coordinates(), axis = 0)
        self.others['x0'] = self.coord_min[0]
        self.others['x1'] = self.coord_max[0] 
        self.others['y0'] = self.coord_min[1]
        self.others['y1'] = self.coord_max[1]
    
    def computeTangent(self):      
        
        self.readMesh()
        sigmaLaw = lambda u: self.lame[0]*nabla_div(u)*df.Identity(2) + 2*self.lame[1]*symgrad(u)
        
        dy = self.mesh.dx # specially for the case of enriched mesh, otherwise it does not work
        vol = df.assemble(df.Constant(1.0)*dy(0)) + df.assemble(df.Constant(1.0)*dy(1))
        
        y = df.SpatialCoordinate(self.mesh)
        Eps = df.Constant(((0.,0.),(0.,0.))) # just placeholder
        
        form = self.multiscaleModel(self.mesh, sigmaLaw, Eps, self.others)
        a,f,bcs,W = form()

        start = timer()        
        A = mp.block_assemble(a)
        if(len(bcs) > 0): 
            bcs.apply(A)
        
        solver = df.PETScLUSolver('superlu')
        sol = mp.BlockFunction(W)
        
        if(self.model == 'lin' or self.model == 'dnn'):
            Vref = self.others['uD'].function_space()
            Mref = Vref.mesh()
            normal = df.FacetNormal(Mref)
            volMref = 4.0
        
        B = np.zeros((2,2))
        
        for i in range(self.nvoigt):
            
            
            start = timer()              
            if(self.model == 'lin' or self.model == 'dnn'):
                self.others['uD'].vector().set_local(self.others['uD{0}_'.format(i)])
            
                B = -feut.Integral(df.outer(self.others['uD'],normal), Mref.ds, (2,2))/volMref
                T = feut.affineTransformationExpression(np.zeros(2),B, Mref) # ignore a, since the basis is already translated
                self.others['uD'].vector().set_local(self.others['uD'].vector().get_local()[:] + 
                                                     df.interpolate(T,Vref).vector().get_local()[:])
                
                
            Eps.assign(df.Constant(mscm.macro_strain(i) - B))   
        
            F = mp.block_assemble(f)
            if(len(bcs) > 0):
                bcs.apply(F)
        
            solver.solve(A,sol.block_vector(), F)    
            sol.block_vector().block_function().apply("to subfunctions")
            
            sig_mu = sigmaLaw(df.dot(Eps,y) + sol[0])
            sigma_hom =  sum([Integral(sig_mu, dy(i), (2,2)) for i in [0,1]])/vol
            
            self.Chom_[:,i] = sigma_hom.flatten()[[0,3,1]]
            
            end = timer()
            print('time in solving system', end - start) # Time in seconds
        
        self.getTangent = self.getTangent_ # from the second run onwards, just returns  
        
        return self.Chom_
        

