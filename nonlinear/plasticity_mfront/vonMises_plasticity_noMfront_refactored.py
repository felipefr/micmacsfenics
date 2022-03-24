from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
parameters["form_compiler"]["representation"] = 'quadrature'
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

ppos = lambda x: (x+abs(x))/2.

def eps(v):
    e = sym(grad(v))
    return as_tensor([[e[0, 0], e[0, 1], 0],
                      [e[0, 1], e[1, 1], 0],
                      [0, 0, 0]])


def as_3D_tensor(X):
    return as_tensor([[X[0], X[3], 0],
                      [X[3], X[1], 0],
                      [0, 0, X[2]]])

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

class plasticityModel:
    
    def sigma(self,eps_el):
        pass
    
    def tangent(self, e):
        pass
    
    def createInternalVariables(self, W, W0):
        pass
    
    def update_alpha(self,deps, old_sig, old_p):
        pass
    
    def project_var(self,AA, labels):
        for label in labels: 
            local_project(AA[label], self.alpha[label].function_space(), self.alpha[label])

class vonMisesPlasticity(plasticityModel):
    
    def __init__(self,E,nu,sig0,Et):
        E = Constant(E)
        nu = Constant(nu)
        self.lmbda = E*nu/(1+nu)/(1-2*nu)
        Et = Constant(Et)  # tangent modulus
        
        self.mu = E/2./(1+nu)
        self.sig0 = Constant(sig0)  # yield strength
        self.H = E*Et/(E-Et)  # hardening modulus
    
    def createInternalVariables(self, W, W0):
        self.sig = Function(W)
        self.sig_old = Function(W)
        self.n_elas = Function(W)
        self.beta = Function(W0)
        self.dp = Function(W0)
        self.p = Function(W0, name="Cumulative plastic strain")

        self.alpha = {'sig': self.sig, 'n_elas' : self.n_elas, 'beta' : self.beta, 'dp' : self.dp}    

    def sigma(self,eps_el):
        return self.lmbda*tr(eps_el)*Identity(3) + 2*self.mu*eps_el
    
    
    def tangent(self, e):
        aAux = 3*self.mu*(3*self.mu/(3*self.mu+self.H)-self.beta)
        N_elas = as_3D_tensor(self.n_elas)

        return self.sigma(e) - aAux*inner(N_elas, e)*N_elas-2*self.mu*self.beta*dev(e)


    def update_alpha(self,deps, old_sig, old_p):
        H, sig0, mu = self.H, self.sig0,self.mu 
        
        sig_n = as_3D_tensor(old_sig)
        sig_elas = sig_n + model.sigma(deps)
        s = dev(sig_elas)
        sig_eq = sqrt(3/2.*inner(s, s))
        f_elas = sig_eq - sig0 - H*old_p
        dp = ppos(f_elas)/(3*mu+H)
        n_elas = s/sig_eq*ppos(f_elas)/f_elas
        beta = 3*mu*dp/sig_eq
        new_sig = sig_elas-beta*s
        
        return  {'sig' : as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
                'n_elas' : as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]), \
                 'beta' : beta, 'dp' : dp}    




# elastic parameters

E = 70e3
nu = 0.3
sig0 = 250.
Et = E/100.

model = vonMisesPlasticity(E,nu,sig0,Et)

Re, Ri = 1.3, 1.   # external/internal radius
mesh = Mesh("thick_cylinder.xml")
facets = MeshFunction("size_t", mesh, "thick_cylinder_facet_region.xml")
ds = Measure('ds')[facets]

deg_u = 2
deg_stress = 2
V = VectorFunctionSpace(mesh, "CG", deg_u)
We = VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
W = FunctionSpace(mesh, We)
W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W0 = FunctionSpace(mesh, W0e)

model.createInternalVariables(W, W0)
u = Function(V, name="Total displacement")
du = Function(V, name="Iteration correction")
Du = Function(V, name="Current increment")
v = TrialFunction(V)
u_ = TestFunction(V)

bc = [DirichletBC(V.sub(1), 0, facets, 1), DirichletBC(V.sub(0), 0, facets, 3)]

n = FacetNormal(mesh)
q_lim = float(2/sqrt(3)*ln(Re/Ri)*sig0)
loading = Expression("-q*t", q=q_lim, t=0, degree=2)

def F_ext(v):
    return loading*dot(n, v)*ds(4)

metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
dxm = dx(metadata=metadata)

# alpha_ = update_sigma(deps, sig_old, p)

a_Newton = inner(eps(v), model.tangent(eps(u_)))*dxm
res = -inner(eps(u_), as_3D_tensor(model.sig))*dxm + F_ext(u_)

file_results = XDMFFile("plasticity_results.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
P0 = FunctionSpace(mesh, "DG", 0)
p_avg = Function(P0, name="Plastic strain")

Nitermax, tol = 200, 1e-8  # parameters of the Newton-Raphson procedure
Nincr = 20
load_steps = np.linspace(0, 1.1, Nincr+1)[1:]**0.5
results = np.zeros((Nincr+1, 2))
for (i, t) in enumerate(load_steps):
    loading.t = t
    A, Res = assemble_system(a_Newton, res, bc)
    nRes0 = Res.norm("l2")
    nRes = nRes0
    Du.vector().set_local(np.zeros(V.dim()))
    
    print("Increment:", str(i+1))
    niter = 0
    while nRes/nRes0 > tol and niter < Nitermax:
        solve(A, du.vector(), Res, "mumps")
        Du.assign(Du+du)
        deps = eps(Du)
        alpha_ = model.update_alpha(deps, model.sig_old, model.p)
        model.project_var(alpha_,['sig','n_elas','beta'])
        A, Res = assemble_system(a_Newton, res, bc)
        nRes = Res.norm("l2")
        print("    Residual:", nRes)
        niter += 1
        
    u.assign(u+Du)
    model.sig_old.assign(model.sig)
    model.project_var(alpha_,['dp'])
    model.p.assign(model.p+ model.dp)


    file_results.write(u, t)
    p_avg.assign(project(model.p, P0))
    file_results.write(p_avg, t)
    results[i+1, :] = (u(Ri, 0)[0], t)


# plt.plot(results[:, 0], results[:, 1], "-o")
# plt.xlabel("Displacement of inner boundary")
# plt.ylabel(r"Applied pressure $q/q_{lim}$")
# plt.show()
