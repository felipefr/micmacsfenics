from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from ddfenics.fenics.enriched_mesh import EnrichedMesh 

from timeit import default_timer as timer

parameters["form_compiler"]["representation"] = 'quadrature'
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

ppos = lambda x: (x+abs(x))/2.

halfsqrt2 = 0.5*np.sqrt(2)


def eps_(v):
    return sym(grad(v))
    
def tensor2mandel(X):
    return as_vector([X[0,0], X[1,1], halfsqrt2*(X[0,1] + X[1,0])])
                      
def mandel2tensor(X):
    return as_tensor([[X[0], halfsqrt2*X[2]],
                      [halfsqrt2*X[2], X[1]]])
      
def tr_mandel(X):
    return X[0] + X[1]


Id_mandel = as_vector([1.0, 1.0, 0.0])
      
def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dxm
    b_proj = inner(v, v_)*dxm
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

class materialModel:
    
    def sigma(self,eps_el):
        pass
    
    def tangent(self, e):
        pass
    
    def createInternalVariables(self, W, W0):
        pass
    
    def update_alpha(self,deps, old_sig, old_p):
        pass
    
    def project_var(self,AA):
        for label in AA.keys(): 
            local_project(AA[label], self.varInt[label].function_space(), self.varInt[label])

class hyperlasticityModel(materialModel):
    
    def __init__(self,E,nu, alpha):
        self.lamb = Constant(E*nu/(1+nu)/(1-2*nu))
        self.mu = Constant(E/2./(1+nu))
        self.alpha = Constant(alpha)
        
    def createInternalVariables(self, W, W0):
        self.sig = Function(W)
        self.eps = Function(W)
        self.tre2 = Function(W0)
        self.ee = Function(W0)
    
        self.varInt = {'tre2': self.tre2, 'ee' : self.ee, 'eps' : self.eps,  'sig' : self.sig} 

    def sigma(self, lamb_, mu_, eps): # elastic (I dont know why for the moment) # in mandel format
        return lamb_*tr_mandel(eps)*Id_mandel + 2*mu_*eps
    
    def epseps_e(self, de):
        return inner(self.eps, de)*self.eps

    
    def tangent(self, de):
        lamb_ = self.lamb*( 1 + 3*self.alpha*self.tre2)
        mu_ = self.mu*( 1 + self.alpha*self.ee ) 
        
        de_mandel = tensor2mandel(de)
        
        return self.sigma(lamb_, mu_, de_mandel)  + 4*self.mu*self.alpha*self.epseps_e(de_mandel)

    def update_alpha(self, eps_new):
        
        ee = inner(eps_new,eps_new)
        tre2 = tr_mandel(eps_new)**2.0
        
        lamb_ = self.lamb*( 1 + self.alpha*tre2)
        mu_ = self.mu*( 1 + self.alpha*ee ) 
        
        alpha_new = {'tre2': tre2, 'ee' : ee, 'eps' : eps_new, 'sig': self.sigma(lamb_, mu_, eps_new)}
        self.project_var(alpha_new)


class hyperlasticityModel_simple(materialModel):
    
    def __init__(self,E,nu, alpha):
        self.lamb = Constant(E*nu/(1+nu)/(1-2*nu))
        self.mu = Constant(E/2./(1+nu))
        self.alpha = Constant(alpha)
        
    def createInternalVariables(self, W, W0):
        self.sig = Function(W)
        self.eps = Function(W)
    
        self.varInt = {'eps' : self.eps,  'sig' : self.sig} 

    def sigma(self, lamb_, mu_, eps): # elastic (I dont know why for the moment) # in mandel format
        return lamb_*tr_mandel(eps)*Id_mandel + 2*mu_*eps
    
    def epseps_e(self, de):
        return inner(self.eps, de)*self.eps

    
    def tangent(self, de):
        ee = inner(self.eps, self.eps)
        tre2 = tr_mandel(self.eps)**2.0
        
        lamb_ = self.lamb*( 1 + 3*self.alpha*tre2)
        mu_ = self.mu*( 1 + self.alpha*ee ) 
        
        de_mandel = tensor2mandel(de)
        
        return self.sigma(lamb_, mu_, de_mandel)  + 4*self.mu*self.alpha*self.epseps_e(de_mandel)

    def update_alpha(self, eps_new):
        
        ee = inner(eps_new,eps_new)
        tre2 = tr_mandel(eps_new)**2.0
        
        lamb_ = self.lamb*( 1 + self.alpha*tre2)
        mu_ = self.mu*( 1 + self.alpha*ee ) 
        
        alpha_new = {'eps' : eps_new, 'sig': self.sigma(lamb_, mu_, eps_new)}
        self.project_var(alpha_new)
            
# elastic parameters

E = 100.0
nu = 0.3
alpha = 200.0
ty = 5.0

model = hyperlasticityModel(E, nu, alpha)

mesh = EnrichedMesh("./meshes/mesh_40.xdmf")


start = timer()

clampedBndFlag = 2 
LoadBndFlag = 1 
traction = Constant((0.0,ty ))
    
deg_u = 1
deg_stress = 1
V = VectorFunctionSpace(mesh, "CG", deg_u)
We = VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=3, quad_scheme='default')
W = FunctionSpace(mesh, We)
W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W0 = FunctionSpace(mesh, W0e)

bcL = DirichletBC(V, Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
bc = [bcL]

def F_ext(v):
    return inner(traction, v)*mesh.ds(LoadBndFlag)

model.createInternalVariables(W, W0)
u = Function(V, name="Total displacement")
du = Function(V, name="Iteration correction")
v = TestFunction(V)
u_ = TrialFunction(V)


metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
dxm = dx(metadata=metadata)

# alpha_ = update_sigma(deps, sig_old, p)

a_Newton = inner(tensor2mandel(eps_(u_)), model.tangent(eps_(v)))*dxm
res = -inner(tensor2mandel(eps_(v)), model.sig)*dxm + F_ext(v)

file_results = XDMFFile("cook.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

Nitermax, tol = 10, 1e-8  # parameters of the Newton-Raphson procedure

A, Res = assemble_system(a_Newton, res, bc)
nRes0 = Res.norm("l2")
nRes = nRes0
du.vector().set_local(np.zeros(V.dim()))
u.vector().set_local(np.zeros(V.dim()))

niter = 0
while nRes/nRes0 > tol and niter < Nitermax:
    solve(A, du.vector(), Res, "mumps")
    u.assign(u + du)
    model.update_alpha(tensor2mandel(eps_(u)))
    A, Res = assemble_system(a_Newton, res, bc)
    nRes = Res.norm("l2")
    print(" Residual:", nRes)
    niter += 1
    


file_results.write(u, 0.0)

end = timer()
print(end - start)


# plt.plot(results[:, 0], results[:, 1], "-o")
# plt.xlabel("Displacement of inner boundary")
# plt.ylabel(r"Applied pressure $q/q_{lim}$")
# plt.show()
